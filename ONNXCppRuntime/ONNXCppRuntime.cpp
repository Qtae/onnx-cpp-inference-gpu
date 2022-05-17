#include <iostream>
#include <fstream>
#include <string>
#include <numeric>
#include <time.h>

#include "onnxruntime_cxx_api.h"
#include "opencv2/opencv.hpp"


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

int main()
{
	bool MONITOR = true;
	std::string modelFilepath = "";
	std::string instanceName = "BF Segmentation Inference";

	Ort::SessionOptions sessionOptions;

	OrtCUDAProviderOptions cuda_options;
	cuda_options.device_id = 0;
	cuda_options.gpu_mem_limit = static_cast<int>(4 * 1024 * 1024 * 1024);
	cuda_options.arena_extend_strategy = 1;
	cuda_options.do_copy_in_default_stream = 1;

	sessionOptions.AppendExecutionProvider_CUDA(cuda_options);
	sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

	Ort::Env env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, instanceName.c_str());
	Ort::Session session(env, L"D:/QTAE/ONNXCppRuntime/onnxModels/stitch.onnx", sessionOptions);
	Ort::AllocatorWithDefaultOptions allocator;

	size_t numInputNodes = session.GetInputCount();
	size_t numOutputNodes = session.GetOutputCount();

	const char* inputName = session.GetInputName(0, allocator);
	Ort::TypeInfo inputTypeInfo = session.GetInputTypeInfo(0);
	auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
	ONNXTensorElementDataType inputType = inputTensorInfo.GetElementType();
	std::vector<int64_t> inputDims = inputTensorInfo.GetShape();
	inputDims[0] = 1;

	const char* outputName = session.GetOutputName(0, allocator);
	Ort::TypeInfo outputTypeInfo = session.GetOutputTypeInfo(0);
	auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();
	ONNXTensorElementDataType outputType = outputTensorInfo.GetElementType();
	std::vector<int64_t> outputDims = outputTensorInfo.GetShape();
	outputDims[0] = 1;


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

	cv::Mat img1 = cv::imread("D:/QTAE/ONNXCppRuntime/Image/stitch/01.jpg", cv::IMREAD_GRAYSCALE);
	cv::Mat img2 = cv::imread("D:/QTAE/ONNXCppRuntime/Image/stitch/02.jpg", cv::IMREAD_GRAYSCALE);
	cv::Mat img3 = cv::imread("D:/QTAE/ONNXCppRuntime/Image/stitch/03.jpg", cv::IMREAD_GRAYSCALE);
	cv::Mat img4 = cv::imread("D:/QTAE/ONNXCppRuntime/Image/stitch/04.jpg", cv::IMREAD_GRAYSCALE);
	cv::Mat img1_f(640, 640, CV_32FC1);
	cv::Mat img2_f(640, 640, CV_32FC1);
	cv::Mat img3_f(640, 640, CV_32FC1);
	cv::Mat img4_f(640, 640, CV_32FC1);
	img1.convertTo(img1_f, CV_32FC1);
	img2.convertTo(img2_f, CV_32FC1);
	img3.convertTo(img3_f, CV_32FC1);
	img4.convertTo(img4_f, CV_32FC1);
	std::vector<cv::Mat> images;
	images.push_back(img1);
	images.push_back(img2);
	images.push_back(img3);
	images.push_back(img4);
	cv::Mat input_blob;
	cv::dnn::blobFromImages(images, input_blob, 1.0, cv::Size(640, 640));
	size_t inputTensorSize = vectorProduct(inputDims);
	std::vector<float> inputTensorValues(inputTensorSize * 4);
	inputTensorValues.assign(input_blob.begin<float>(), input_blob.end<float>());

	size_t outputTensorSize = vectorProduct(outputDims);
	std::vector<float> outputTensorValues(outputTensorSize * 4);

	std::vector<const char*> inputNames{ inputName };
	std::vector<const char*> outputNames{ outputName };
	std::vector<Ort::Value> inputTensors;
	std::vector<Ort::Value> outputTensors;

	Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
	inputTensors.push_back(Ort::Value::CreateTensor<float>(memoryInfo, inputTensorValues.data(), inputTensorSize, inputDims.data(), inputDims.size()));
	outputTensors.push_back(Ort::Value::CreateTensor<float>(memoryInfo, outputTensorValues.data(), outputTensorSize, outputDims.data(), outputDims.size()));

	time_t start = clock();
	session.Run(Ort::RunOptions{ nullptr },
		inputNames.data(), inputTensors.data(), 1,
		outputNames.data(), outputTensors.data(), 1);
	time_t end = clock();

	if (MONITOR)
		std::cout << "Inference Time: " << (double)(end - start) << " ms" << std::endl;

	//start = clock();
	//session.Run(Ort::RunOptions{ nullptr },
	//	inputNames.data(), inputTensors.data(), 1,
	//	outputNames.data(), outputTensors.data(), 1);
	//end = clock();
	//
	//if (MONITOR)
	//	std::cout << "Inference Time: " << (double)(end - start) << " ms" << std::endl;

	//outputTensorValues.at(i);
	return 0;
}