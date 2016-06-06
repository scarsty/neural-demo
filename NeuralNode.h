#pragma once
#include <vector>
#include <map>
#include <functional>
#include <string>
#include "ActiveFunctions.h"

//键
class NeuralBond
{
public:
	double weight = 0;
	double learnSpeed = 1;
	class NeuralNode* startNode;
	class NeuralNode* endNode;

	void updateWeight(double learnSpeed);
};


//神经元的类型，包含隐藏，输入，输出
typedef enum 
{
	Hidden,
	Input,
	Output,
	Const,
} NeuralNodeType;

//神经元，神经节点
class NeuralNode
{
public:
	NeuralNode();
	virtual ~NeuralNode();

	NeuralNodeType type;
	std::string tag;
	int id;

	std::map<NeuralNode*, NeuralBond*> prevBonds;  //这里好像只保存weight就行了
	std::map<NeuralNode*, NeuralBond*> nextBonds;  //next实际为prev的镜像，start和end相同时保存的是同一个指针

	NeuralBond*& getPrevBond(NeuralNode* node) { return prevBonds[node]; };
	NeuralBond*& getNextBond(NeuralNode* node) { return nextBonds[node]; };  

	//double outputValue;
	//double inputValue;
	//double expect;
	
	//多组数据
	//没有下标安全检查，使用需慎重！
	static int dataAmount;
	std::vector<double> outputValues;
	std::vector<double> inputValues;
	std::vector<double> expects;
	std::vector<double> deltas;

	void setExpect(double x, int i = -1);  //设置期待值，一般仅用于输出节点
	void setInput(double x, int i = -1);   //设置输入值，可以用于常数节点
	void setOutput(double x, int i = -1);  //设置输出值，一般仅用于输入节点
	double getOutput(int i = -1);

	void collectInputValue();
	void activeOutputValue();
	void active();

	//feedback是active的导数
	std::function<double(double)> activeFunction = ActiveFunctions::sigmoid;
	std::function<double(double)> dactiveFunction = ActiveFunctions::dsigmoid;

	void setFunctions(std::function<double(double)> _active, std::function<double(double)> _dactive);

	static void connect(NeuralNode* start, NeuralNode* end, double w = 0);
	void connectStart(NeuralNode* node, double w = 0);
	void connectEnd(NeuralNode* node, double w = 0);

	//double delta;
	void updateOneDelta();  //无用

	void setVectorValue(std::vector<double>& vec, double x = -1) { for (auto& v : vec) v = x; }

	void setDataAmount(int n);
	int getDataAmount() { return dataAmount; }
	void updateDelta();
	void backPropagate(double learnSpeed = 0.5);

	bool actived, backPropageted;

};

