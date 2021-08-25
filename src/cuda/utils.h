#pragma once


#include <string>
#include <vector>
#include <iostream>

struct Args{
    enum MODE {test, simulation};
    
    std::vector<int> ss;
    std::vector<double> ps;
    MODE mode;
    
    int s;
    double p;
    unsigned long long random_seed = 0;
    int simulation_num = 1;
    bool verbose = false;
    bool per = true;
    
    Args(int argc, char* argv[]){
        for(int iter=0;iter<argc;iter++){
            std::string act_param = argv[iter];
            if(act_param=="--p"){
                int size = std::stoi(argv[++iter]);
                ps.resize(size);
                ps.front() = std::stod(argv[++iter]);
                ps.back() = std::stod(argv[++iter]);
                
                double delta = (size < 2 ? 0 : (ps.back()-ps.front())/(size-1));
                for(int i=1;i<size-1;i++){
                    ps[i] = ps[i-1]+delta;
                }
                p = ps.front();
                //p = std::stod(argv[++iter]);
            }
            else if(act_param=="--s"){
                int size = std::stoi(argv[++iter]);
                ss.resize(size);
                ss.front() = std::stoi(argv[++iter]);
                ss.back() = std::stoi(argv[++iter]);
                
                double delta = (size < 2 ? 0 : (ss.back()-ss.front())/(size-1));
                for(int i=1;i<size-1;i++){
                    ss[i] = ss[i-1]+delta;
                }
                s = ss.front();                
            }
            else if(act_param=="--s_nonlin"){
                int size = std::stoi(argv[++iter]);
                ss.resize(size);
                
                for(int i=0;i<size;i++){
                    ss[i] = std::stoi(argv[++iter]);
                }
                s = ss.front();                
            }
            else if(act_param=="--p_nonlin"){
                int size = std::stoi(argv[++iter]);
                ps.resize(size);
                
                for(int i=0;i<size;i++){
                    ps[i] = std::stof(argv[++iter]);
                    //std::cout<<std::to_string(ps[i])<<" "<<argv[iter]<<std::endl;
                }
                p = ps.front();                
            }
            else if(act_param=="--seed") random_seed = std::stoull(argv[++iter]);
            else if(act_param=="--sim_num") simulation_num = std::stoi(argv[++iter]);
            else if(act_param=="--verbose"){ verbose=true; ++iter;}
            else if(act_param=="--per"){
                act_param = argv[++iter];
                if(act_param == "True") per = true;
                else if(act_param == "False") per = false;
                else std::cout<<"--per param not set properly\n";
            }
            else if(act_param=="--mode"){
                act_param = argv[++iter];
                if(act_param == "test") mode = test;
                else if(act_param == "simulation") mode = simulation;
            }
        }
        std::cout<<">>> Params: p="<<p<<" seed="<<random_seed<<" verbose="<<(verbose?"True":"False")<<"\n"; 
    }
};