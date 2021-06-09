#ifndef HANDLER_H
#define HANDLER_H
#include <iostream>
#include "stdafx.h"

using namespace std;
using namespace web;
using namespace http;
using namespace utility;
using namespace http::experimental::listener;


class handler
{
    public:
        handler(utility::string_t url);

        pplx::task<void>open(){return m_listener.open();}
        pplx::task<void>close(){return m_listener.close();}
    static void init_bag();
    protected:

    private:
        void handle_post(http_request message);
        http_listener m_listener;
};

#endif // HANDLER_H
