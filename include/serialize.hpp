#ifndef SERIALIZE_H
#define SERIALIZE_H

#include <cereal/archives/portable_binary.hpp>
#include <cereal/types/vector.hpp>

struct Road_User{
    float latitude;
    float longitude;
    uint8_t speed;
    uint8_t orientation;
    uint8_t category;

    template<class Archive>
    void serialize(Archive & archive)
    {
        archive( latitude, longitude, speed, orientation, category ); 
    }

};

struct Message{
    uint32_t cam_idx;
    uint64_t t_stamp_ms;
    uint16_t num_objects;
    std::vector<Road_User> objects;
    
    template<class Archive>
    void serialize(Archive & archive)
    {
        archive( cam_idx, t_stamp_ms, num_objects, objects ); 
    }
};




#endif