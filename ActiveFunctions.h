#pragma once
#include <math.h>
//#include "NeuralNode.h"

class ActiveFunctionPair 
{
public:
	virtual double active(double input);
	virtual double _active(double output);
};

class ActiveFunctions
{
public:
	ActiveFunctions();
	virtual ~ActiveFunctions();
	//static xoid setFunctions(class NeuralNode* node, std::function<double(double)> activeFunction, std::function<double(double)> feedbackFunction);

	double c = 0, f = 0;
	static double sigmoid(double x) { return 1.0 / (1 + exp(-x)); }
	static double dsigmoid(double x) { double a = 1 + exp(-x); return exp(-x) / (a*a); }
	static double linear(double x) { return x; }
	static double dlinear(double x) { return 1; }
	static double exp1(double x) { return exp(x); }
	static double dexp1(double x) { return exp(x); }
	static double tanh1(double x) { return tanh(x); }
	static double dtanh1(double x) { return 1 / cosh(x) / cosh(x); }

	static double sign1(double x) { return x > 0 ? 1 : -1; }
	static double dsign1(double x) { return 1; }

	static double is(double x) { return x > 0.5 ? 1 : 0; }
	static double dis(double x) { return 1; }

};

