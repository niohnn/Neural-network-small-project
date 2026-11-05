#include <loss/BinaryCrossEntropy.h>

#include <MathUtils.h>

#include <algorithm>
#include <cmath>

namespace dnn {

    double BinaryCrossEntropy::getCostValue(
        const std::vector<double>& predictedOutput,
        const std::vector<double>& actualOutput
    ) const {
        double cost = 0.0;
        for(std::size_t i=0; i<predictedOutput.size();i++){
            double p= clampProbability(predictedOutput[i]);
            
            cost += actualOutput[i]*std::log(p)+(1.0-actualOutput[i])*std::log(1.0-p);
        }
        return -cost/predictedOutput.size();
    }

    //Pi is log(clampProbability(predictedOutput[i])), Qi is 1-same log Ri is 1-actualOutput[i] and n is the number of predictions 
    // n here is represented by the loop precitedoutput.site()
    double BinaryCrossEntropy::getAccuracyValue(
        const std::vector<double>& predictedOutput,
        const std::vector<double>& actualOutput
    ) const {
        double correct = 0.0;
        for (std::size_t i =0; i<predictedOutput.size();i++){
            double prediction = predictedOutput[i]>0.5 ? 1.0 : 0.0;
            if(prediction==actualOutput[i]){
                correct++;
            }
        }
        return correct/predictedOutput.size();
        //prediction strictly greater than 0.5 up to 1.0 otherwise 0.0 compare each binary prediction with actual output divide number of matches by n 
    }
    std ::vector <double> BinaryCrossEntropy::getGradient(
        const std::vector <double>& predictedOutput,
        const std ::vector<double>& actualOutput    
    ) const {
        std::vector<double>gradient(predictedOutput.size(),0.0);
        for(std::size_t i = 0 ; i < predictedOutput.size();i++){
            double p = clampProbability(predictedOutput[i]);
            gradient[i]= -(actualOutput[i]/p-(1.0-actualOutput[i])/(1.0-p))/predictedOutput.size();
        }
        return gradient;
    }
std::string BinaryCrossEntropy::getName() const {
    return "binary_cross_entropy";
}

}
