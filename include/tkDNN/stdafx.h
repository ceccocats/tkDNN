#ifndef STDAFX_H_INCLUDED
#define STDAFX_H_INCLUDED
#define BOOST_LOG_DYN_LINK 1

#pragma once

#include <string>
#include <vector>
#include <algorithm>
#include <sstream>
#include <iostream>
#include <fstream>
#include <random>

#ifdef _WIN32
#define NOMINMAX
#include <Windows.h>
#else
# include <sys/time.h>
#endif

#include "cpprest/json.h"
#include "cpprest/http_listener.h"
#include "cpprest/uri.h"
#include "cpprest/asyncrt_utils.h"
#include "cpprest/json.h"
#include "cpprest/filestream.h"
#include "cpprest/containerstream.h"
#include "cpprest/producerconsumerstream.h"

#include <boost/log/core.hpp>
#include <boost/log/trivial.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/utility/setup/file.hpp>
#include <boost/log/utility/setup/common_attributes.hpp>
#include <boost/asio/ip/host_name.hpp>

#pragma warning ( push )
#pragma warning ( disable : 4457 )
#pragma warning ( pop )
#include <locale>
#include <ctime>
#endif // STDAFX_H_INCLUDED
