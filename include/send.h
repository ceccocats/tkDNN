#ifndef SEND_H
#define SEND_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <sys/socket.h> //socket
#include <arpa/inet.h>  //inet_addr
#include <unistd.h>     //write

struct obj_coords
{
    float LAT;
    float LONG;
    float cl;
};


char *serialize_coords(struct obj_coords *c, int obj_n, int CAM_IDX)
{

    char *buffer = (char *)malloc(100000 * sizeof(char));

    struct timeval tv;
    gettimeofday(&tv, NULL);
    unsigned long long t_stamp_ms = (unsigned long long)(tv.tv_sec) * 1000 + (unsigned long long)(tv.tv_usec) / 1000;

    sprintf(buffer, "m %lld %d %d ", t_stamp_ms, CAM_IDX, obj_n);
    int i;
    for (i = 0; i < obj_n; i++)
        sprintf(buffer + strlen(buffer), "%.9f %.9f %.0f ", c[i].LAT, c[i].LONG, c[i].cl);

    //printf("%s\n", buffer);

    return buffer;
}

struct obj_coords *deserialize_coords(char *buffer, int *obj_n)
{

    int cam_id;
    unsigned long long t_stamp_ms;
    char type_of_m;

    int consumed_chars = 0;
    char shifted_chars[100000];

    sscanf(buffer, "%c %lld %d %d ", &type_of_m, &t_stamp_ms, &cam_id, obj_n);
    sprintf(shifted_chars, "%c %lld %d %d ", type_of_m, t_stamp_ms, cam_id, *obj_n);
    consumed_chars += strlen(shifted_chars);
    //printf("%c %lld %d %d\n", type_of_m, t_stamp_ms, cam_id, *obj_n);

    struct obj_coords *c = (struct obj_coords *)malloc(*obj_n * sizeof(struct obj_coords));

    int i;
    for (i = 0; i < *obj_n; i++)
    {
        sscanf(buffer + consumed_chars, "%f %f %f ", &c[i].LAT, &c[i].LONG, &c[i].cl);
        sprintf(shifted_chars, "%.9f %.9f %.0f ", c[i].LAT, c[i].LONG, c[i].cl);
        consumed_chars += strlen(shifted_chars);
        //printf("%Lf %f %f \n", c[i].LAT, c[i].LONG, c[i].cl);
    }

    return c;
}

int map_class_coco_to_voc(int coco_class)
{
    switch (coco_class)
    {
    case 0:
        return 14; //person
    case 1:
        return 1; //bicycle
    case 2:
        return 6; //car
    case 3:
        return 13; //motorkbike
    case 5:
        return 5; //bus
    }
    return -1;
}

void convert_coords(struct obj_coords *coords, int i, int x, int y, int detected_class,float * proj_matrix)
{

    

    float obj_x = x, obj_y = y, obj_z = 1;
    float tmp_z = 0;

    coords[i].LAT = proj_matrix[0] * obj_x + proj_matrix[1] * obj_y + proj_matrix[2] * obj_z;
    coords[i].LONG = proj_matrix[3] * obj_x + proj_matrix[4] * obj_y + proj_matrix[5] * obj_z;
    tmp_z = proj_matrix[6] * obj_x + proj_matrix[7] * obj_y + proj_matrix[8] * obj_z;
    
    if(tmp_z != 0.0)
    {
        coords[i].LAT = coords[i].LAT / tmp_z;
        coords[i].LONG = coords[i].LONG / tmp_z;
        //printf("lat: %f, long %f\n", coords[i].LAT, coords[i].LONG);
    }
    else
        printf("Division by 0 (tmp_z)\n");
    coords[i].cl = map_class_coco_to_voc(detected_class);
}

void read_projection_matrix(float * proj_matrix, int &proj_matrix_read, char* path)
{
    FILE *fp;
    char *line = NULL;
    size_t len = 0;
    ssize_t read;

    fp = fopen(path, "r");
    if (fp == NULL)
        exit(EXIT_FAILURE);

    int i = 0;
    while ((read = getline(&line, &len, fp)) != -1)
    {
        if (3 == sscanf(line, "%f %f %f", &proj_matrix[i*3+0], &proj_matrix[i*3+1], &proj_matrix[i*3+2]))
        {
            i++;
            proj_matrix_read = 1;
        }
    }
    free(line);
    fclose(fp);
}

int open_socket(char *ip, int &sock, int &socket_opened)
{
    struct sockaddr_in server;
    sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock == -1)
    {
        printf("Could not create socket");
    }
    puts("Socket created");    

    server.sin_addr.s_addr = inet_addr(ip);
    server.sin_family = AF_INET;
    server.sin_port = htons(8888);

    /*connect to remote server*/
    if (connect(sock, (struct sockaddr *)&server, sizeof(server)) < 0)
    {
        perror("connect failed. Error");
        socket_opened = 0;
        return 0;
    }
    puts("Connected\n");
    socket_opened = 1;
    return 1;
}

int send_client_dummy(struct obj_coords *coords, int n_coords, int &sock, int &socket_opened, int CAM_IDX)
{
    /*serialize coords*/
    char *message = serialize_coords(coords, n_coords, CAM_IDX);

    /*open socket if not already opened*/
    if (socket_opened == 0)
    {
        int res = open_socket("127.0.0.1",sock, socket_opened);
        if(res)
            printf("Socket opened!\n");
        else
        {
            printf("Problem: socket NOT opened!\n");
            return 0;
        }
    }

    /*send message to server*/
    if (send(sock, message, strlen(message), 0) < 0)
    {
        puts("Send failed");
        socket_opened = 0;
    }

    free(message);
    //close(sock);
    return 1;
}

#endif /*SEND_H*/