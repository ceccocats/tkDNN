#define _XOPEN_SOURCE 700

//
// a single-threaded, multi client(using select), debug webserver - streaming out mjpg.
//  on win, _WIN32 has to be defined, must link against ws2_32.lib (socks on linux are for free)
//

#include <cstdio>
#include <vector>
#include <iostream>
#include <algorithm>
#include <memory>
#include <mutex>
#include <thread>
#include <atomic>
#include <ctime>
using std::cerr;
using std::endl;

//
// socket related abstractions:
//
#ifdef _WIN32
#ifndef USE_CMAKE_LIBS
#pragma comment(lib, "ws2_32.lib")
#endif
#define WIN32_LEAN_AND_MEAN
#define _WINSOCK_DEPRECATED_NO_WARNINGS
#include <windows.h>
#include <winsock2.h>
#include <ws2tcpip.h>
#include "gettimeofday.h"
#define PORT        unsigned long
#define ADDRPOINTER   int*
struct _INIT_W32DATA
{
    WSADATA w;
    _INIT_W32DATA() { WSAStartup(MAKEWORD(2, 1), &w); }
} _init_once;

// Graceful closes will first close their output channels and then wait for the peer
// on the other side of the connection to close its output channels. When both sides are done telling
// each other they won,t be sending any more data (i.e., closing output channels),
// the connection can be closed fully, with no risk of reset.
static int close_socket(SOCKET s) {
    int close_output = ::shutdown(s, 1); // 0 close input, 1 close output, 2 close both
    char *buf = (char *)calloc(1024, sizeof(char));
    ::recv(s, buf, 1024, 0);
    free(buf);
    int close_input = ::shutdown(s, 0);
    int result = ::closesocket(s);
    cerr << "Close socket: out = " << close_output << ", in = " << close_input << " \n";
    return result;
}
#else   // _WIN32 - else: nix
#include "darkunistd.h"
#include <fcntl.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <signal.h>
#define PORT        unsigned short
#define SOCKET    int
#define HOSTENT  struct hostent
#define SOCKADDR    struct sockaddr
#define SOCKADDR_IN  struct sockaddr_in
#define ADDRPOINTER  unsigned int*
#define INVALID_SOCKET -1
#define SOCKET_ERROR   -1
struct _IGNORE_PIPE_SIGNAL
{
    struct sigaction new_actn, old_actn;
    _IGNORE_PIPE_SIGNAL() {
        new_actn.sa_handler = SIG_IGN;  // ignore the broken pipe signal
        sigemptyset(&new_actn.sa_mask);
        new_actn.sa_flags = 0;
        sigaction(SIGPIPE, &new_actn, &old_actn);
        // sigaction (SIGPIPE, &old_actn, NULL); // - to restore the previous signal handling
    }
} _init_once;

static int close_socket(SOCKET s) {
    int close_output = ::shutdown(s, 1); // 0 close input, 1 close output, 2 close both
    char *buf = (char *)calloc(1024, sizeof(char));
    ::recv(s, buf, 1024, 0);
    free(buf);
    int close_input = ::shutdown(s, 0);
    int result = close(s);
    std::cerr << "Close socket: out = " << close_output << ", in = " << close_input << " \n";
    return result;
}
#endif // _WIN32

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc_c.h>
#ifndef CV_VERSION_EPOCH
#include <opencv2/videoio/videoio.hpp>
#endif

using namespace cv;



class MJPG_sender
{
    SOCKET sock;
    SOCKET maxfd;
    fd_set master;
    int timeout; // master sock timeout, shutdown after timeout usec.
    int quality; // jpeg compression [1..100]
    int close_all_sockets;

    int _write(int sock, char const*const s, int len)
    {
        if (len < 1) { len = strlen(s); }
        return ::send(sock, s, len, 0);
    }

public:

    MJPG_sender(int port = 0, int _timeout = 400000, int _quality = 30)
        : sock(INVALID_SOCKET)
        , timeout(_timeout)
        , quality(_quality)
    {
        close_all_sockets = 0;
        FD_ZERO(&master);
        if (port)
            open(port);
    }

    ~MJPG_sender()
    {
        close_all();
        release();
    }

    bool release()
    {
        if (sock != INVALID_SOCKET)
            ::shutdown(sock, 2);
        sock = (INVALID_SOCKET);
        return false;
    }

    void close_all()
    {
        close_all_sockets = 1;
        cv::Mat tmp(cv::Size(10, 10), CV_8UC3);
        write(tmp);
    }

    bool open(int port)
    {
        sock = ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);

        SOCKADDR_IN address;
        address.sin_addr.s_addr = INADDR_ANY;
        address.sin_family = AF_INET;
        address.sin_port = htons(port);    // ::htons(port);
        int reuse = 1;
        if (setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, (const char*)&reuse, sizeof(reuse)) < 0)
            cerr << "setsockopt(SO_REUSEADDR) failed" << endl;

        // Non-blocking sockets
        // Windows: ioctlsocket() and FIONBIO
        // Linux: fcntl() and O_NONBLOCK
#ifdef WIN32
        unsigned long i_mode = 1;
        int result = ioctlsocket(sock, FIONBIO, &i_mode);
        if (result != NO_ERROR) {
            std::cerr << "ioctlsocket(FIONBIO) failed with error: " << result << std::endl;
        }
#else // WIN32
        int flags = fcntl(sock, F_GETFL, 0);
        fcntl(sock, F_SETFL, flags | O_NONBLOCK);
#endif // WIN32

#ifdef SO_REUSEPORT
        if (setsockopt(sock, SOL_SOCKET, SO_REUSEPORT, (const char*)&reuse, sizeof(reuse)) < 0)
            cerr << "setsockopt(SO_REUSEPORT) failed" << endl;
#endif
        if (::bind(sock, (SOCKADDR*)&address, sizeof(SOCKADDR_IN)) == SOCKET_ERROR)
        {
            cerr << "error MJPG_sender: couldn't bind sock " << sock << " to port " << port << "!" << endl;
            return release();
        }
        if (::listen(sock, 10) == SOCKET_ERROR)
        {
            cerr << "error MJPG_sender: couldn't listen on sock " << sock << " on port " << port << " !" << endl;
            return release();
        }
        FD_ZERO(&master);
        FD_SET(sock, &master);
        maxfd = sock;
        return true;
    }

    bool isOpened()
    {
        return sock != INVALID_SOCKET;
    }

    bool write(cv::Mat frame)
    {
        fd_set rread = master;
        struct timeval select_timeout = { 0, 0 };
        struct timeval socket_timeout = { 0, timeout };
        if (::select(maxfd + 1, &rread, NULL, NULL, &select_timeout) <= 0)
            return true; // nothing broken, there's just noone listening

        std::vector<uchar> outbuf;
        std::vector<int> params;
        params.push_back(IMWRITE_JPEG_QUALITY);
        params.push_back(quality);
        cv::imencode(".jpg", frame, outbuf, params);  //REMOVED FOR COMPATIBILITY
        // https://docs.opencv.org/3.4/d4/da8/group__imgcodecs.html#ga292d81be8d76901bff7988d18d2b42ac
        //std::cerr << "cv::imencode call disabled!" << std::endl;
        int outlen = static_cast<int>(outbuf.size());

#ifdef _WIN32
        for (unsigned i = 0; i<rread.fd_count; i++)
        {
            int addrlen = sizeof(SOCKADDR);
            SOCKET s = rread.fd_array[i];    // fd_set on win is an array, while ...
#else
        for (int s = 0; s <= maxfd; s++)
        {
            socklen_t addrlen = sizeof(SOCKADDR);
            if (!FD_ISSET(s, &rread))      // ... on linux it's a bitmask ;)
                continue;
#endif
            if (s == sock) // request on master socket, accept and send main header.
            {
                SOCKADDR_IN address = { 0 };
                SOCKET      client = ::accept(sock, (SOCKADDR*)&address, &addrlen);
                if (client == SOCKET_ERROR)
                {
                    cerr << "error MJPG_sender: couldn't accept connection on sock " << sock << " !" << endl;
                    return false;
                }
                if (setsockopt(client, SOL_SOCKET, SO_RCVTIMEO, (char *)&socket_timeout, sizeof(socket_timeout)) < 0) {
                    cerr << "error MJPG_sender: SO_RCVTIMEO setsockopt failed\n";
                }
                if (setsockopt(client, SOL_SOCKET, SO_SNDTIMEO, (char *)&socket_timeout, sizeof(socket_timeout)) < 0) {
                    cerr << "error MJPG_sender: SO_SNDTIMEO setsockopt failed\n";
                }
                maxfd = (maxfd>client ? maxfd : client);
                FD_SET(client, &master);
                _write(client, "HTTP/1.0 200 OK\r\n", 0);
                _write(client,
                    "Server: Mozarella/2.2\r\n"
                    "Accept-Range: bytes\r\n"
                    "Connection: close\r\n"
                    "Max-Age: 0\r\n"
                    "Expires: 0\r\n"
                    "Cache-Control: no-cache, private\r\n"
                    "Pragma: no-cache\r\n"
                    "Content-Type: multipart/x-mixed-replace; boundary=mjpegstream\r\n"
                    "\r\n", 0);
                cerr << "MJPG_sender: new client " << client << endl;
            }
            else // existing client, just stream pix
            {
                if (close_all_sockets) {
                    int result = close_socket(s);
                    cerr << "MJPG_sender: close clinet: " << result << " \n";
                    continue;
                }

                char head[400];
                sprintf(head, "--mjpegstream\r\nContent-Type: image/jpeg\r\nContent-Length: %zu\r\n\r\n", outlen);
                _write(s, head, 0);
                int n = _write(s, (char*)(&outbuf[0]), outlen);
                cerr << "known client: " << s << ", sent = " << n << ", must be sent outlen = " << outlen << endl;
                if (n < (int)outlen)
                {
                    cerr << "MJPG_sender: kill client " << s << endl;
                    //::shutdown(s, 2);
                    close_socket(s);
                    FD_CLR(s, &master);
                }
            }
        }
        if (close_all_sockets) {
            int result = close_socket(sock);
            cerr << "MJPG_sender: close acceptor: " << result << " \n\n";
        }
        return true;
    }
};
// ----------------------------------------

static std::mutex mtx_mjpeg;

//struct mat_cv : cv::Mat { int a[0]; };

void send_mjpeg(cv::Mat mat, int port, int timeout, int quality)
{
    try {
        std::lock_guard<std::mutex> lock(mtx_mjpeg);
        static MJPG_sender wri(port, timeout, quality);
        //cv::Mat mat = cv::cvarrToMat(ipl);
        wri.write(mat);
        std::cout << " MJPEG-stream sent. \n";
    }
    catch (...) {
        cerr << " Error in send_mjpeg() function \n";
    }
}
// ----------------------------------------

std::string get_system_frame_time_string()
{
    std::time_t t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    static std::mutex mtx;
    std::lock_guard<std::mutex> lock(mtx);
    struct tm *tmp_buf = localtime(&t);
    char buff[256];
    std::strftime(buff, 256, "%A %F %T", tmp_buf);
    std::string system_frame_time = buff;
    return system_frame_time;
}
// ----------------------------------------


/*#ifdef __CYGWIN__
int send_http_post_request(char *http_post_host, int server_port, const char *videosource,
    detection *dets, int nboxes, int classes, char **names, long long int frame_id, int ext_output, int timeout)
{
    std::cerr << " send_http_post_request() isn't implemented \n";
    return 0;
}
#else   //  __CYGWIN__*/
#ifndef   NI_MAXHOST
#define   NI_MAXHOST 1025
#endif

#ifndef   NI_NUMERICHOST
#define NI_NUMERICHOST  0x02
#endif
