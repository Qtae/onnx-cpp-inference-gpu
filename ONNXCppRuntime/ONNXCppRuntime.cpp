#include <iostream>
#include <fstream>
#include <string>
#include <numeric>
#include <time.h>

#include "onnxruntime_cxx_api.h"
#include "opencv2/opencv.hpp"
#include "Eigen/Eigen"


struct DetectionResult
{
	int x;
	int y;
	int w;
	int h;
	float Objectness;
	int BestClass;
	float Score;
};

template <typename T>
T vectorProduct(const std::vector<T>& v)
{
	return accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v)
{
	os << "[";
	for (int i = 0; i < v.size(); ++i)
	{
		os << v[i];
		if (i != v.size() - 1)
		{
			os << ", ";
		}
	}
	os << "]";
	return os;
}

std::ostream& operator<<(std::ostream& os, const ONNXTensorElementDataType& type)
{
	switch (type)
	{
	case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED:
		os << "undefined";
		break;
	case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
		os << "float";
		break;
	case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
		os << "uint8_t";
		break;
	case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
		os << "int8_t";
		break;
	case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
		os << "uint16_t";
		break;
	case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
		os << "int16_t";
		break;
	case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
		os << "int32_t";
		break;
	case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
		os << "int64_t";
		break;
	case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
		os << "std::string";
		break;
	case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
		os << "bool";
		break;
	case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
		os << "float16";
		break;
	case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
		os << "double";
		break;
	case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
		os << "uint32_t";
		break;
	case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
		os << "uint64_t";
		break;
	case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:
		os << "float real + float imaginary";
		break;
		case ONNXTensorElementDataType::
		ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:
			os << "double real + float imaginary";
			break;
		case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
			os << "bfloat16";
			break;
		default:
			break;
	}

	return os;
}

std::vector<std::string> readLabels(std::string& labelFilepath)
{
	std::vector<std::string> labels;
	std::string line;
	std::ifstream fp(labelFilepath);
	while (std::getline(fp, line))
	{
		labels.push_back(line);
	}
	return labels;
}

bool CompareScore(DetectionResult x, DetectionResult y)
{
	if (x.Score * x.Objectness > y.Score * y.Objectness) return true;
	else return false;
}

float CalculateIOU(DetectionResult box1, DetectionResult box2)
{
	int maxX = std::max(box1.x - (box1.w / 2), box2.x - (box2.w / 2));
	int maxY = std::max(box1.y - (box1.h / 2), box2.y - (box2.h / 2));
	int minX = std::min(box1.x + (box1.w / 2), box2.x + (box2.w / 2));
	int minY = std::min(box1.y + (box1.h / 2), box2.y + (box2.h / 2));
	int overlapWidth = ((minX - maxX + 1) > 0) ? (minX - maxX + 1) : 0;
	int overlapHeight = ((minY - maxY + 1) > 0) ? (minY - maxY + 1) : 0;
	int overlapArea = overlapWidth * overlapHeight;
	int box1Area = box1.h * box1.w;
	int box2Area = box2.h * box2.w;
	return float(overlapArea) / float(box1Area + box2Area - overlapArea);
}

void ApplyScoreThreshold(std::vector<DetectionResult>& detRes, float scoreThresh)
{
	for (std::vector<DetectionResult>::iterator it = detRes.begin(); it != detRes.end();)
	{
		if ((*it).Objectness * (*it).Score < scoreThresh) it = detRes.erase(it);
		else it++;
	}
	return;
}

void DoNMS(std::vector<DetectionResult>& detRes, float iouThresh, float scoreThresh, int clsNum)
{
	if (detRes.empty()) return;
	std::vector<std::vector<DetectionResult>> detResOfEachClass;
	for (int s = 0; s < clsNum; ++s)
	{
		std::vector<DetectionResult> tmp;
		for (std::vector<DetectionResult>::iterator it = detRes.begin(); it != detRes.end();)
		{
			if ((*it).BestClass == s)
			{
				tmp.push_back(*it);
				it = detRes.erase(it);
			}
			else it++;
		}
		ApplyScoreThreshold(tmp, scoreThresh);
		sort(tmp.begin(), tmp.end(), CompareScore);//sort the candidate boxes by confidence
		detResOfEachClass.push_back(tmp);
	}

	for (int s = 0; s < clsNum; ++s)
	{
		for (int i = 0; i < detResOfEachClass[s].size(); i++)
		{
			if (detResOfEachClass[s][i].Score > 0)
			{
				for (int j = i + 1; j < detResOfEachClass[s].size(); j++)
				{
					if (detResOfEachClass[s][j].Score > 0)
					{
						float iou = CalculateIOU(detResOfEachClass[s][i], detResOfEachClass[s][j]);//calculate the orthogonal ratio
						if (iou > iouThresh) detResOfEachClass[s][j].Score = 0;
					}
				}
			}
		}
		for (std::vector<DetectionResult>::iterator it = detResOfEachClass[s].begin(); it != detResOfEachClass[s].end(); ++it)
		{
			if ((*it).Score != 0) detRes.push_back(*it);
		}
	}
	return;
}

int main()
{
	//Ort::Float16_t
	bool MONITOR = true;
	std::string modelFilepath = "";
	std::string instanceName = "BF Segmentation Inference";

	Ort::SessionOptions* sessionOptions;
	sessionOptions = new Ort::SessionOptions();

	OrtCUDAProviderOptions* cuda_options = new OrtCUDAProviderOptions();
	cuda_options->device_id = 0;
	cuda_options->gpu_mem_limit = static_cast<int>(4 * 1024 * 1024 * 1024);
	cuda_options->arena_extend_strategy = 1;
	cuda_options->do_copy_in_default_stream = 1;

	sessionOptions->AppendExecutionProvider_CUDA(*cuda_options);
	sessionOptions->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

	Ort::Env* env = new Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, instanceName.c_str());
	Ort::Session* session;
	time_t start1 = clock();
	session = new Ort::Session(*env, L"D:/QTAE/ONNXCppRuntime/onnxModels/bf_q.onnx", *sessionOptions);
	time_t end1 = clock();
	if (MONITOR)
		std::cout << "Load Time: " << (double)(end1 - start1) << " ms" << std::endl;

	//Ort::Session session(env, L"D:/QTAE/ONNXCppRuntime/onnxModels/bf.onnx", sessionOptions);
	//Ort::Session session(env, L"D:/QTAE/ONNXCppRuntime/onnxModels/stitch.onnx", sessionOptions);
	Ort::AllocatorWithDefaultOptions allocator;

	size_t numInputNodes = session->GetInputCount();
	size_t numOutputNodes = session->GetOutputCount();

	const char* inputName = session->GetInputName(0, allocator);
	Ort::TypeInfo inputTypeInfo = session->GetInputTypeInfo(0);
	auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
	ONNXTensorElementDataType inputType = inputTensorInfo.GetElementType();
	std::vector<int64_t> inputDims = inputTensorInfo.GetShape();
	inputDims[0] = 4;

	const char* outputName = session->GetOutputName(0, allocator);
	Ort::TypeInfo outputTypeInfo = session->GetOutputTypeInfo(0);
	auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();
	ONNXTensorElementDataType outputType = outputTensorInfo.GetElementType();
	std::vector<int64_t> outputDims = outputTensorInfo.GetShape();
	outputDims[0] = 4;

	if (MONITOR)
	{
		std::cout << "Number of Input Nodes: " << numInputNodes << std::endl;
		std::cout << "Number of Output Nodes: " << numOutputNodes << std::endl;
		std::cout << "Input Name: " << inputName << std::endl;
		std::cout << "Input Type: " << inputType << std::endl;
		std::cout << "Input Dimensions: " << inputDims << std::endl;
		std::cout << "Output Name: " << outputName << std::endl;
		std::cout << "Output Type: " << outputType << std::endl;
		std::cout << "Output Dimensions: " << outputDims << std::endl;
	}

	cv::Mat img1 = cv::imread("D:/QTAE/ONNXCppRuntime/Image/bf/01.bmp", cv::IMREAD_GRAYSCALE);
	cv::Mat img2 = cv::imread("D:/QTAE/ONNXCppRuntime/Image/bf/02.bmp", cv::IMREAD_GRAYSCALE);
	cv::Mat img3 = cv::imread("D:/QTAE/ONNXCppRuntime/Image/bf/03.bmp", cv::IMREAD_GRAYSCALE);
	cv::Mat img4 = cv::imread("D:/QTAE/ONNXCppRuntime/Image/bf/04.bmp", cv::IMREAD_GRAYSCALE);
	cv::Mat img1_f(640, 640, CV_32FC1);
	cv::Mat img2_f(640, 640, CV_32FC1);
	cv::Mat img3_f(640, 640, CV_32FC1);
	cv::Mat img4_f(640, 640, CV_32FC1);
	img1.convertTo(img1_f, CV_32FC1);
	img2.convertTo(img2_f, CV_32FC1);
	img3.convertTo(img3_f, CV_32FC1);
	img4.convertTo(img4_f, CV_32FC1);

	size_t inputTensorSize = vectorProduct(inputDims);
	std::vector<Ort::Float16_t> inputTensorValues;
	std::vector<float> tmp(inputTensorSize / 4);
	std::vector<Ort::Float16_t> tmp2(inputTensorSize / 4);
	memcpy(&tmp[0], img1_f.data, (inputTensorSize / 4) * sizeof(float));
	for (int i = 0; i < tmp.size(); ++i)
	{
		tmp[i] /= 255.0;
		tmp2[i] = Ort::Float16_t(Eigen::half(tmp[i]).x);
	}
	inputTensorValues.insert(inputTensorValues.end(), tmp2.begin(), tmp2.end());
	memcpy(&tmp[0], img2_f.data, (inputTensorSize / 4) * sizeof(float));
	for (int i = 0; i < tmp.size(); ++i)
	{
		tmp[i] /= 255.0;
		tmp2[i] = Ort::Float16_t(Eigen::half(tmp[i]).x);
	}
	inputTensorValues.insert(inputTensorValues.end(), tmp2.begin(), tmp2.end());
	memcpy(&tmp[0], img3_f.data, (inputTensorSize / 4) * sizeof(float));
	for (int i = 0; i < tmp.size(); ++i)
	{
		tmp[i] /= 255.0;
		tmp2[i] = Ort::Float16_t(Eigen::half(tmp[i]).x);
	}
	inputTensorValues.insert(inputTensorValues.end(), tmp2.begin(), tmp2.end());
	memcpy(&tmp[0], img4_f.data, (inputTensorSize / 4) * sizeof(float));
	for (int i = 0; i < tmp.size(); ++i)
	{
		tmp[i] /= 255.0;
		tmp2[i] = Ort::Float16_t(Eigen::half(tmp[i]).x);
	}
	inputTensorValues.insert(inputTensorValues.end(), tmp2.begin(), tmp2.end());
	
	//inputTensorValues.assign(img1_f.begin<float>(), img1_f.end<float>());

	size_t outputTensorSize = vectorProduct(outputDims);
	std::vector<Ort::Float16_t> outputTensorValues(outputTensorSize);

	std::vector<const char*> inputNames;
	std::vector<const char*> outputNames;
	inputNames.push_back(inputName);
	outputNames.push_back(outputName);
	std::vector<Ort::Value> inputTensors;
	std::vector<Ort::Value> outputTensors;

	Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
	inputTensors.push_back(Ort::Value::CreateTensor<Ort::Float16_t>(memoryInfo, inputTensorValues.data(), inputTensorSize, inputDims.data(), inputDims.size()));
	outputTensors.push_back(Ort::Value::CreateTensor<Ort::Float16_t>(memoryInfo, outputTensorValues.data(), outputTensorSize, outputDims.data(), outputDims.size()));

	time_t start = clock();
	session->Run(Ort::RunOptions{ nullptr },
		inputNames.data(), inputTensors.data(), 1,
		outputNames.data(), outputTensors.data(), 1);
	time_t end = clock();

	if (MONITOR)
		std::cout << "Inference Time: " << (double)(end - start) << " ms" << std::endl;
	///* Segmentation
	unsigned char* outputput = new unsigned char[640 * 640];
	cv::Mat output1(640, 640, CV_8UC1);
	cv::Mat output2(640, 640, CV_8UC1);
	cv::Mat output3(640, 640, CV_8UC1);
	cv::Mat output4(640, 640, CV_8UC1);
	for (int imgIdx = 0; imgIdx < 4; ++imgIdx)
	{
		for (int i = 0; i < inputTensorSize / 4; ++i)
		{
			float score1 = outputTensorValues[imgIdx * (inputTensorSize / 2) + 2 * i];
			float score2 = outputTensorValues[imgIdx * (inputTensorSize / 2) + 2 * i + 1];
			//float expSum = exp(score1) + exp(score2);
			//score1 = exp(score1) / expSum;
			//score2 = exp(score2) / expSum;
			if (score1 > score2) outputput[i] = 0;
			else outputput[i] = 255;
		}
		if (imgIdx == 0) memcpy(output1.data, outputput, 640 * 640 * sizeof(unsigned char));
		if (imgIdx == 1) memcpy(output2.data, outputput, 640 * 640 * sizeof(unsigned char));
		if (imgIdx == 2) memcpy(output3.data, outputput, 640 * 640 * sizeof(unsigned char));
		if (imgIdx == 3) memcpy(output4.data, outputput, 640 * 640 * sizeof(unsigned char));

	}
	//*/

	/*
	start = clock();
	session->Run(Ort::RunOptions{ nullptr },
		inputNames.data(), inputTensors.data(), 1,
		outputNames.data(), outputTensors.data(), 1);
	end = clock();
	
	if (MONITOR)
		std::cout << "Inference Time: " << (double)(end - start) << " ms" << std::endl;

	start = clock();
	session->Run(Ort::RunOptions{ nullptr },
		inputNames.data(), inputTensors.data(), 1,
		outputNames.data(), outputTensors.data(), 1);
	end = clock();

	if (MONITOR)
		std::cout << "Inference Time: " << (double)(end - start) << " ms" << std::endl;

	start = clock();
	session->Run(Ort::RunOptions{ nullptr },
		inputNames.data(), inputTensors.data(), 1,
		outputNames.data(), outputTensors.data(), 1);
	end = clock();

	if (MONITOR)
		std::cout << "Inference Time: " << (double)(end - start) << " ms" << std::endl;

	start = clock();
	session->Run(Ort::RunOptions{ nullptr },
		inputNames.data(), inputTensors.data(), 1,
		outputNames.data(), outputTensors.data(), 1);
	end = clock();

	if (MONITOR)
		std::cout << "Inference Time: " << (double)(end - start) << " ms" << std::endl;

	start = clock();
	session->Run(Ort::RunOptions{ nullptr },
		inputNames.data(), inputTensors.data(), 1,
		outputNames.data(), outputTensors.data(), 1);
	end = clock();

	if (MONITOR)
		std::cout << "Inference Time: " << (double)(end - start) << " ms" << std::endl;

	start = clock();
	session->Run(Ort::RunOptions{ nullptr },
		inputNames.data(), inputTensors.data(), 1,
		outputNames.data(), outputTensors.data(), 1);
	end = clock();

	if (MONITOR)
		std::cout << "Inference Time: " << (double)(end - start) << " ms" << std::endl;

	start = clock();
	session->Run(Ort::RunOptions{ nullptr },
		inputNames.data(), inputTensors.data(), 1,
		outputNames.data(), outputTensors.data(), 1);
	end = clock();

	if (MONITOR)
		std::cout << "Inference Time: " << (double)(end - start) << " ms" << std::endl;

	start = clock();
	session->Run(Ort::RunOptions{ nullptr },
		inputNames.data(), inputTensors.data(), 1,
		outputNames.data(), outputTensors.data(), 1);
	end = clock();

	if (MONITOR)
		std::cout << "Inference Time: " << (double)(end - start) << " ms" << std::endl;

	start = clock();
	session->Run(Ort::RunOptions{ nullptr },
		inputNames.data(), inputTensors.data(), 1,
		outputNames.data(), outputTensors.data(), 1);
	end = clock();

	if (MONITOR)
		std::cout << "Inference Time: " << (double)(end - start) << " ms" << std::endl;
	*/

	/* Detection
	std::vector<std::vector<DetectionResult>> Result;
	int gridX = 20;
	int gridY = 20;
	int anchorNum = 3;
	int clsNum = 3;
	int varNum = 12;
	for (int imgIdx = 0; imgIdx < 4; ++imgIdx)
	{
		std::vector<DetectionResult> tmpRes;

		for (int grdXIdx = 0; grdXIdx < gridX; ++grdXIdx)
		{
			for (int grdYIdx = 0; grdYIdx < gridY; ++grdYIdx)
			{
				for (int ancIdx = 0; ancIdx < anchorNum; ++ancIdx)
				{
					DetectionResult detRes;
					detRes.x = (int)outputTensorValues[imgIdx * gridX * gridY * anchorNum * varNum +
						grdXIdx * gridY * anchorNum * varNum +
						grdYIdx * anchorNum * varNum +
						ancIdx * varNum +
						0];
					detRes.y = (int)outputTensorValues[imgIdx * gridX * gridY * anchorNum * varNum +
						grdXIdx * gridY * anchorNum * varNum +
						grdYIdx * anchorNum * varNum +
						ancIdx * varNum +
						1];
					detRes.w = (int)outputTensorValues[imgIdx * gridX * gridY * anchorNum * varNum +
						grdXIdx * gridY * anchorNum * varNum +
						grdYIdx * anchorNum * varNum +
						ancIdx * varNum +
						2];
					detRes.h = (int)outputTensorValues[imgIdx * gridX * gridY * anchorNum * varNum +
						grdXIdx * gridY * anchorNum * varNum +
						grdYIdx * anchorNum * varNum +
						ancIdx * varNum +
						3];
					detRes.Objectness = outputTensorValues[imgIdx * gridX * gridY * anchorNum * varNum +
						grdXIdx * gridY * anchorNum * varNum +
						grdYIdx * anchorNum * varNum +
						ancIdx * varNum +
						4];
					int bestCls = -1;
					float score = 0.;
					for (int clsIdx = 0; clsIdx < clsNum; ++clsIdx)
					{
						float currScore = outputTensorValues[imgIdx * gridX * gridY * anchorNum * varNum +
							grdXIdx * gridY * anchorNum * varNum +
							grdYIdx * anchorNum * varNum +
							ancIdx * varNum +
							5 + clsIdx];
						if (currScore >= score)
						{
							bestCls = clsIdx;
							score = currScore;
						}
					}
					detRes.BestClass = bestCls;
					detRes.Score = score;
					tmpRes.push_back(detRes);
				}
			}
		}
		DoNMS(tmpRes, 0.3, 0.25, clsNum);
		Result.push_back(tmpRes);
	}
	*/

	return 0;
}