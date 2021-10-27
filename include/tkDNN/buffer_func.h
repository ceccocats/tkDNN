#ifndef BUFFER_FUNC_H
#define BUFFER_FUNC_H

namespace tk { namespace dnn {

template<typename T> void writeBUF(char*& buffer, const T& val)
{
    *reinterpret_cast<T*>(buffer) = val;
    buffer += sizeof(T);
}

template<typename T> T readBUF(const char*& buffer)
{
    T val = *reinterpret_cast<const T*>(buffer);
    buffer += sizeof(T);
    return val;
}
}}

#endif // BUFFER_FUNC_H