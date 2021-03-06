#ifndef TSP2D_ENV_H
#define TSP2D_ENV_H

#include "i_env.h"
#include <string>
extern int sign;

class Tsp2dEnv : public IEnv
{
public:

    Tsp2dEnv(double _norm, const std::string record_filename); // avrech

    virtual void s0(std::shared_ptr<Graph>  _g) override;

    virtual double step(int a) override;

    virtual int randomAction() override;

    virtual bool isTerminal() override;

    double add_node(int new_node);
    std::string rec_filename; // avrech
    std::set<int> partial_set;
    std::vector<int> avail_list;
};

#endif