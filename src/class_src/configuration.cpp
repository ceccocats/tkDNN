#include "configuration.h"

/* Read the configuration file. The yaml file contains the information about the network, 
 * the map and the cameras.
**/
void readCamerasParametersYaml(const std::string &camerasParams, Parameters_t *par)
{

    std::string cam, tmp;
    YAML::Node config = YAML::LoadFile(camerasParams);
    tmp = config["weights"].as<std::string>();//config["lastLogin"].as<DateTime>()
    par->net = (char *) malloc((strlen(tmp.c_str())+1) * sizeof(char));
    strcpy(par->net, tmp.c_str());
    par->net[strlen(tmp.c_str())] = '\0';
    // std::cout<<"- "<<par->net<<std::endl;  
    tmp = config["tif_map"].as<std::string>();
    par->tiffile = (char *) malloc((strlen(tmp.c_str())+1) * sizeof(char));
    strcpy(par->tiffile, tmp.c_str());
    par->tiffile[strlen(tmp.c_str())] = '\0';
    // std::cout<<"- "<<par->tiffile<<std::endl;
        
    //read cameras infomrations
    for (int i = 0; i<par->n_cameras; i++)
    {
        cam = std::to_string(par->cameras[i].CAM_IDX);
        // std::cout<<"read "<<cam<<std::endl;
        tmp = config[cam]["input_stream"].as<std::string>();
        par->cameras[i].input = (char *) malloc((strlen(tmp.c_str())+1) * sizeof(char));
        strcpy(par->cameras[i].input, tmp.c_str());
        par->cameras[i].input[strlen(tmp.c_str())] = '\0';
        // std::cout<<"- "<<par->cameras[i].input<<std::endl;
        tmp = config[cam]["pmatrix"].as<std::string>();
        par->cameras[i].pmatrix = (char *) malloc((strlen(tmp.c_str())+1) * sizeof(char));
        strcpy(par->cameras[i].pmatrix, tmp.c_str());
        par->cameras[i].pmatrix[strlen(tmp.c_str())] = '\0';
        // std::cout<<"- "<<par->cameras[i].pmatrix<<std::endl;
        tmp = config[cam]["maskfile"].as<std::string>();
        par->cameras[i].maskfile = (char *) malloc((strlen(tmp.c_str())+1) * sizeof(char));
        strcpy(par->cameras[i].maskfile, tmp.c_str());
        par->cameras[i].maskfile[strlen(tmp.c_str())] = '\0';
        // std::cout<<"- "<<par->cameras[i].maskfile<<std::endl;
        tmp = config[cam]["cameraCalib"].as<std::string>();
        par->cameras[i].cameraCalib = (char *) malloc((strlen(tmp.c_str())+1) * sizeof(char));
        strcpy(par->cameras[i].cameraCalib, tmp.c_str());
        par->cameras[i].cameraCalib[strlen(tmp.c_str())] = '\0';
        // std::cout<<"- "<<par->cameras[i].cameraCalib<<std::endl;
        tmp = config[cam]["maskFileOrient"].as<std::string>();
        par->cameras[i].maskFileOrient = (char *) malloc((strlen(tmp.c_str())+1) * sizeof(char));
        strcpy(par->cameras[i].maskFileOrient, tmp.c_str());
        par->cameras[i].maskFileOrient[strlen(tmp.c_str())] = '\0';
        // std::cout<<"- "<<par->cameras[i].maskFileOrient<<std::endl;
    }
}

/* Read the parameters from the command line and configuration file.
**/
bool read_parameters(int argc, char *argv[], Parameters_t *par)
{
    bool no_params = false;
    std::string params;// = "../../prova.yaml";
    par->n_cameras = 0;
    int opt;
    char *help = "Yolo3 demo\nCommand:\n -i\tencrypted parameters file\n -n\tnumer of cameras\n \tlist of camera numbers (see -n)\n \tlist of flags for the visualization (see -n)\n\n";
    // first read n_cameras parameter and/or help option 
    while((opt = getopt(argc, argv, ":n:i:h")) != -1)
    {
        switch(opt)
        {
            case 'n': //number of cameras
                std::cout<<"number of cameras"<<std::endl;
                par->n_cameras = atoi(optarg);
                break;
            case 'h':
                std::cout<<"help"<<std::endl;
                std::cout<<help<<std::endl;
                return false;
            case 'i':
                    std::cout<<"input parameters file"<<std::endl;
                    params = optarg;
                    std::cout<<"file: "<<params<<std::endl;
                    break;
            case ':':
                printf("option needs a value\n");
                no_params = true;
                par->n_cameras = 1;
                break;
            case '?':
                printf("unknown option: %c\n", optopt);
                break;
        }
    }
    par->cameras = (Camera_t *) malloc(par->n_cameras * sizeof(Camera_t)); 
    bool *to_show = (bool *) malloc(par->n_cameras * sizeof(bool)); 
    if(no_params) 
    {
        par->net = "yolo3_coco4.rt";
        par->tiffile = "../demo/demo/data/map_b.tif";
        par->cameras[0].CAM_IDX = 20936;
        par->cameras[0].input = (char *)"../demo/demo/data/single_ped_2.mp4";
        par->cameras[0].pmatrix = (char *)"../demo/demo/data/pmundist.txt";
        par->cameras[0].maskfile = (char *)"../demo/demo/data/mask36.jpg";
        par->cameras[0].cameraCalib = (char *)"../demo/demo/data/calib36.params";
        par->cameras[0].maskFileOrient = (char *)"../demo/demo/data/mask_orient/6315_mask_orient.jpg";
        par->cameras[0].to_show = true;
        to_show[0] = true;
    }
    else
    {    
        int i = 0;
        for(; optind < argc; optind++)
        {
            if(i==par->n_cameras)
                break;
            printf("extra arguments: %s\n", argv[optind]);
            par->cameras[i].CAM_IDX = atoi(argv[optind]);
            i++;
        }
        i=0;
        for(; optind < argc; optind++)
        {
            if(i==par->n_cameras)
                break;
            printf("extra arguments: %s\n", argv[optind]);
            par->cameras[i].to_show = atoi(argv[optind]); //only one camera can be shown
            to_show[i] = par->cameras[i].to_show;
            i++;
        }
        // TODO: now only one camera can be visualized
        int check_visualization=0;
        for(int i = 0; i<par->n_cameras; i++)
            check_visualization += to_show[i];
        if(check_visualization > 1)
            return false;
        //decrypt the input file, save it in tmp directory
        char s[200] = "";
        strcat(s, "openssl enc -aes-256-cbc -d -in ");
        strcat(s, params.c_str());
        strcat(s, " -base64 -md sha1 -out /tmp/decrypt.yaml");
        if(system(s))
        {
            fprintf(stderr, "Error system\n");
            return false;
        }
        // read the file yaml to get the camera parameters
        readCamerasParametersYaml("/tmp/decrypt.yaml", par);
        // delete the file decrypted
        if (system("rm /tmp/decrypt.yaml"))
        {
            fprintf(stderr, "Error system\n");
            return false;
        }
    }
    return true;
}