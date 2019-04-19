/*
    C socket server example, handles multiple clients using threads
*/

#include <stdio.h>
#include <string.h> //strlen
#include <stdlib.h> //strlen
#include <time.h>
#include <sys/socket.h>
#include <arpa/inet.h> //inet_addr
#include <unistd.h>    //write
#include <pthread.h>   //for threading , link with lpthread
#include <semaphore.h>
#include <sys/types.h>
#include <sys/stat.h>
#include "serialize.hpp"
#include <string>

sem_t semaphore;

struct obj_coords
{
    float LAT;
    float LONG;
    float cl;
};

int write_coords_to_file(std::string buffer, FILE *f)
{

    std::istringstream is(buffer);
    cereal::PortableBinaryInputArchive retrieve(is);
    Message m;
    retrieve(m);

    int cam_id = m.cam_idx;
    unsigned long long t_stamp_ms = m.t_stamp_ms;
    int obj_n = m.num_objects;

    struct obj_coords *c = (struct obj_coords *)malloc(obj_n * sizeof(struct obj_coords));

    int i;
    for (i = 0; i < obj_n; i++)
    {
        c[i].LAT = m.objects.at(i).latitude;
        c[i].LONG = m.objects.at(i).longitude;
        c[i].cl = m.objects.at(i).category;
        //m.objects.at(i).speed;
        //m.objects.at(i).orientation;
    }

    char *to_print = (char *)malloc(100000);
    memset(to_print, 0, 100000);

    /*int obj_n;
    int cam_id;
    unsigned long long t_stamp_ms;
    char type_of_m;

    int consumed_chars = 0;
    char *shifted_chars = (char*)malloc(4000);
    char *to_print = (char*)malloc(100000);
    memset(to_print,0,100000);

    sscanf(buffer, "%c %lld %d %d ", &type_of_m, &t_stamp_ms, &cam_id, &obj_n);
    sprintf(shifted_chars, "%c %lld %d %d ", type_of_m, t_stamp_ms, cam_id, obj_n);
    consumed_chars += strlen(shifted_chars);
    //printf("%c %lld %d %d\n", type_of_m, t_stamp_ms, cam_id, *obj_n);

    struct obj_coords *c = (struct obj_coords *)malloc(obj_n * sizeof(struct obj_coords));

    int i;
    for (i = 0; i < obj_n; i++)
    {
        sscanf(buffer + consumed_chars, "%f %f %f ", &c[i].LAT, &c[i].LONG, &c[i].cl);
        sprintf(shifted_chars, "%.9f %.9f %.0f ", c[i].LAT, c[i].LONG, c[i].cl);
        consumed_chars += strlen(shifted_chars);
        //printf("%Lf %f %f \n", c[i].LAT, c[i].LONG, c[i].cl);
    }*/

    char *command = (char *)malloc(4000);
    for (i = 0; i < obj_n; i++)
    {
        sprintf(command, "%d %lld %.0f %.9f %.9f %f %f\n", cam_id, t_stamp_ms, c[i].cl, c[i].LAT, c[i].LONG, 0.0f, 0.0f);
        strcat(to_print, command);
    }

    printf("%s", to_print);

    fprintf(f, "%s", to_print);
    free(to_print);
    free(command);
    free(c);

    return obj_n;
}

//the thread function
void *connection_handler(void *);

int main(int argc, char *argv[])
{
    const int path_size = 400;
    //const int message_size = 100000;
    char path[path_size];
    //int obj_n;
    const char *basepath = "/tmp/";
    struct stat st = {0};

    sem_init(&semaphore, 0, 1);

    int socket_desc, client_sock, c, *new_sock;
    struct sockaddr_in server, client;

    time_t rawtime;
    time(&rawtime);
    struct tm *tm_struct = localtime(&rawtime);
    int tm_hour = tm_struct->tm_hour;
    int tm_yday = tm_struct->tm_yday;

    sprintf(path, "%s%d/", basepath, tm_yday);
    if (stat(path, &st) == -1)
    {
        mkdir(path, 0700);
    }
    memset(path, 0, path_size);

    sprintf(path, "%s%d/%d/", basepath, tm_yday, tm_hour);
    if (stat(path, &st) == -1)
    {
        mkdir(path, 0700);
    }
    memset(path, 0, path_size);

    //Create socket
    socket_desc = socket(AF_INET, SOCK_STREAM, 0);
    if (socket_desc == -1)
    {
        printf("Could not create socket");
    }
    puts("Socket created");

    //Prepare the sockaddr_in structure
    server.sin_family = AF_INET;
    server.sin_addr.s_addr = INADDR_ANY;
    server.sin_port = htons(8888);

    //Bind
    if (bind(socket_desc, (struct sockaddr *)&server, sizeof(server)) < 0)
    {
        //print the error message
        perror("bind failed. Error");
        return 1;
    }
    puts("bind done");

    //Listen
    listen(socket_desc, 3);

    //Accept and incoming connection
    puts("Waiting for incoming connections...");
    c = sizeof(struct sockaddr_in);
    while ((client_sock = accept(socket_desc, (struct sockaddr *)&client, (socklen_t *)&c)))
    {
        puts("Connection accepted");

        pthread_t sniffer_thread;
        new_sock = (int *)malloc(1);
        *new_sock = client_sock;

        if (pthread_create(&sniffer_thread, NULL, connection_handler, (void *)new_sock) < 0)
        {
            perror("could not create thread");
            return 1;
        }

        //Now join the thread , so that we dont terminate before the thread
        //pthread_join( sniffer_thread , NULL);
        puts("Handler assigned");
    }

    if (client_sock < 0)
    {
        perror("accept failed");
        return 1;
    }

    sem_destroy(&semaphore);

    return 0;
}

/*
 * This will handle connection for each client
 * */
void *connection_handler(void *socket_desc)
{
    time_t rawtime;
    struct tm *tm_struct;
    int tm_hour;
    int tm_yday;
    const int path_size = 400;
    const int message_size = 100000;
    char path[path_size];
    int obj_n;
    const char *basepath = "/tmp/";
    struct stat st = {0};
    FILE *f;
    int mess_hour, mess_min, mess_yday;

    //Get the socket descriptor
    int sock = *(int *)socket_desc;
    int read_size;
    
    void *client_message = (void*)malloc(message_size);

    /* //Send some messages to the client
    message = "Greetings! I am your connection handler\n";
    write(sock, message, strlen(message)); */

    //Receive a message from client
    while ((read_size = recv(sock, client_message, message_size, 0)) > 0)
    {

        std::string s((char *)client_message, message_size);
        //std::cout<<"Message received: "<<s<<std::endl;

        time(&rawtime);
        tm_struct = localtime(&rawtime);
        mess_min = tm_struct->tm_min;
        mess_hour = tm_struct->tm_hour;
        mess_yday = tm_struct->tm_yday;
        if (mess_yday != tm_yday)
        {
            tm_yday = mess_yday;
            sprintf(path, "%s%d/", basepath, tm_yday);
            if (stat(path, &st) == -1)
            {
                mkdir(path, 0700);
            }
            memset(path, 0, path_size);
        }
        if (mess_hour != tm_hour)
        {
            tm_hour = mess_hour;
            sprintf(path, "%s%d/%d/", basepath, tm_yday, tm_hour);
            if (stat(path, &st) == -1)
            {
                mkdir(path, 0700);
            }
            memset(path, 0, path_size);
        }

        sprintf(path, "%s%d/%d/%d.txt", basepath, tm_yday, tm_hour, mess_min);

        //CRITICAL SECTION
        sem_wait(&semaphore);
        f = fopen(path, "a");
        memset(path, 0, path_size);

        obj_n = write_coords_to_file(s, f);

        fclose(f);
        sem_post(&semaphore);
    }

    if (read_size == 0)
    {
        puts("Client disconnected");
        fflush(stdout);
    }
    else if (read_size == -1)
    {
        perror("recv failed");
    }

    //Free the socket pointer
    free(socket_desc);

    free(client_message);

    return 0;
}
