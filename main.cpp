#include <iostream>

#include "stdafx.h"
#include "handler.h"
using namespace std;
using namespace web;
using namespace http;
using namespace utility;
using namespace http::experimental::listener;

namespace logging = boost::log;
namespace keywords = boost::log::keywords;

std::unique_ptr<handler> g_httpHandler;

string get_file_name(string path)
{
    return path.substr(path.find_last_of("/\\")+1);
}

void init_logging()
{
    logging::register_simple_formatter_factory<logging::trivial::severity_level, char>("Severity");

    auto host_name = boost::asio::ip::host_name();
    string logFileName = "server_" + string(host_name) + ".log";

    logging::add_file_log(
        keywords::file_name = "/home/baggageai/log/"+logFileName,
        keywords::format = "BAI-[%LineID%] [%TimeStamp%] [%Severity%] %Message%",
        keywords::auto_flush = true
    );

    logging::core::get()->set_filter
    (
        logging::trivial::severity >= logging::trivial::info
    );

    logging::add_common_attributes();
}

void on_initialize(const string_t& address)
{
    uri_builder uri(address);

    try
    {
        auto addr = uri.to_uri().to_string();
        g_httpHandler = std::unique_ptr<handler>(new handler(addr));
        g_httpHandler->open().wait();

        BOOST_LOG_TRIVIAL(info) << "[" << get_file_name(string(__FILE__)) << "  " << __LINE__ << "] " << "Listening for requests at: "+ string(addr);

        while(true);
    }
    catch (exception const& e)
    {
        BOOST_LOG_TRIVIAL(error) << "[" << get_file_name(string(__FILE__)) << "  " << __LINE__ << "] " << e.what();
        wcout << e.what() << endl;
    }
}

void on_shutdown()
{
    g_httpHandler->close().wait();
    return;
}

#ifdef _WIN32
int wmain(int argc, wchar_t *argv[])
#else
int main(int argc, char *argv[])
#endif
{
    init_logging();
    handler::init_bag();
    utility::string_t port = U("8080");
    if(argc == 2)
    {
        port = argv[1];
    }

    utility::string_t address = U("http://0.0.0.0:");
    address.append(port);

    on_initialize(address);
    return 0;
}
