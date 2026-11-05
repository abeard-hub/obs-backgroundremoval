// SPDX-FileCopyrightText: 2021-2026 Roy Shilkrot <roy.shil@gmail.com>
// SPDX-FileCopyrightText: 2023-2026 Kaito Udagawa <umireon@kaito.tokyo>
//
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef MODELRVM_H
#define MODELRVM_H

#include "Model.hpp"

#include <cmath>
#include <cstring>
#include <string>

/**
 * Robust Video Matting (MobileNetV3).
 *
 * The ONNX model has fully dynamic input shapes and a `downsample_ratio` input: the encoder runs at
 * (input size x ratio) and RVM's deep guided filter refines the alpha back to the full input size.
 *
 * Quality modes:
 *  - Fast:     legacy behaviour, a fixed 320x192 input (aspect ratio ignored), ratio 1.0.
 *  - Balanced: aspect-correct input up to 640 px on the long side, encoder at 320 px.
 *  - High:     aspect-correct input up to 1024 px on the long side, encoder at 384 px.
 *
 * Recurrent state tensors are kept as ORT-owned values whose shapes follow whatever the model
 * returns, so they stay correct for any input size.
 */
class ModelRVM : public ModelBCHW {
private:
	uint32_t inputWidth = 320;
	uint32_t inputHeight = 192;
	float downsampleRatio = 1.0f;

	uint32_t targetLongSide = 640; // 0 = legacy fixed 320x192
	uint32_t encoderLongSide = 320;

	uint32_t lastFrameWidth = 0;
	uint32_t lastFrameHeight = 0;

	size_t ratioInputIndex = 5;

	// r1..r4 returned by the previous run, fed back as r1i..r4i
	std::vector<Ort::Value> recurrent;
	Ort::Value lastAlpha{nullptr};
	bool hasAlpha = false;

	static uint32_t roundTo16(double v)
	{
		const long r = std::lround(v / 16.0) * 16;
		return (uint32_t)std::max(64L, r);
	}

	static int recurrentIndexFromName(const char *name, char suffix)
	{
		// "r1i".."r4i" or "r1o".."r4o"
		if (name && std::strlen(name) == 3 && name[0] == 'r' && name[1] >= '1' && name[1] <= '4' &&
		    name[2] == suffix) {
			return name[1] - '1';
		}
		return -1;
	}

public:
	enum Quality { QUALITY_FAST = 0, QUALITY_BALANCED = 1, QUALITY_HIGH = 2 };

	ModelRVM(/* args */) {}
	~ModelRVM() {}

	void setQuality(int quality)
	{
		uint32_t newTarget = 640, newEncoder = 320;
		if (quality == QUALITY_FAST) {
			newTarget = 0;
			newEncoder = 0;
		} else if (quality == QUALITY_HIGH) {
			newTarget = 1024;
			newEncoder = 384;
		}
		if (newTarget != targetLongSide || newEncoder != encoderLongSide) {
			targetLongSide = newTarget;
			encoderLongSide = newEncoder;
			// force configureForFrame() to recompute on the next frame
			lastFrameWidth = 0;
			lastFrameHeight = 0;
		}
	}

	virtual bool configureForFrame(uint32_t frameWidth, uint32_t frameHeight)
	{
		if (frameWidth == 0 || frameHeight == 0) {
			return false;
		}
		if (frameWidth == lastFrameWidth && frameHeight == lastFrameHeight) {
			return false;
		}
		lastFrameWidth = frameWidth;
		lastFrameHeight = frameHeight;

		uint32_t newWidth = 320, newHeight = 192;
		float newRatio = 1.0f;
		if (targetLongSide > 0) {
			const double frameLong = (double)std::max(frameWidth, frameHeight);
			double longSide = std::min((double)targetLongSide, frameLong); // never upscale
			longSide = std::max(longSide, (double)encoderLongSide);
			const double scale = longSide / frameLong;
			newWidth = roundTo16(frameWidth * scale);
			newHeight = roundTo16(frameHeight * scale);
			newRatio = std::min(1.0f, (float)encoderLongSide / (float)std::max(newWidth, newHeight));
		}

		const bool changed = newWidth != inputWidth || newHeight != inputHeight || newRatio != downsampleRatio;
		inputWidth = newWidth;
		inputHeight = newHeight;
		downsampleRatio = newRatio;
		if (changed) {
			obs_log(LOG_INFO, "RVM input %ux%u, downsample ratio %.3f (frame %ux%u)", inputWidth, inputHeight,
				downsampleRatio, frameWidth, frameHeight);
			resetTemporalState();
		}
		return changed;
	}

	virtual void resetTemporalState()
	{
		recurrent.clear();
		lastAlpha = Ort::Value{nullptr};
		hasAlpha = false;
	}

	virtual void populateInputOutputNames(const std::unique_ptr<Ort::Session> &session,
					      std::vector<Ort::AllocatedStringPtr> &inputNames,
					      std::vector<Ort::AllocatedStringPtr> &outputNames)
	{
		Ort::AllocatorWithDefaultOptions allocator;

		inputNames.clear();
		outputNames.clear();

		for (size_t i = 0; i < session->GetInputCount(); i++) {
			inputNames.push_back(session->GetInputNameAllocated(i, allocator));
			if (std::strcmp(inputNames.back().get(), "downsample_ratio") == 0) {
				ratioInputIndex = i;
			}
		}
		// Output 0 is the foreground colour (fgr), which we don't need
		for (size_t i = 1; i < session->GetOutputCount(); i++) {
			outputNames.push_back(session->GetOutputNameAllocated(i, allocator));
		}
	}

	virtual bool populateInputOutputShapes(const std::unique_ptr<Ort::Session> &session,
					       std::vector<std::vector<int64_t>> &inputDims,
					       std::vector<std::vector<int64_t>> &outputDims)
	{
		inputDims.clear();
		outputDims.clear();

		const size_t inputCount = session->GetInputCount();
		for (size_t i = 0; i < inputCount; i++) {
			if (i == 0) {
				inputDims.push_back({1, 3, (int64_t)inputHeight, (int64_t)inputWidth});
			} else if (i == ratioInputIndex) {
				inputDims.push_back({1});
			} else {
				// initial recurrent state: a 1x1x1x1 zero tensor, as in the reference implementation
				inputDims.push_back({1, 1, 1, 1});
			}
		}

		const size_t outputCount = session->GetOutputCount();
		for (size_t i = 1; i < outputCount; i++) {
			if (i == 1) {
				outputDims.push_back({1, 1, (int64_t)inputHeight, (int64_t)inputWidth});
			} else {
				outputDims.push_back({1, 1, 1, 1}); // placeholder, real outputs are ORT-allocated
			}
		}
		return true;
	}

	virtual void loadInputToTensor(const cv::Mat &preprocessedImage, uint32_t, uint32_t,
				       std::vector<std::vector<float>> &inputTensorValues)
	{
		std::vector<float> &src = inputTensorValues[0];
		const size_t count = preprocessedImage.total() * preprocessedImage.channels();
		if (src.size() == count) {
			// copy in place so the Ort::Value wrapping this buffer stays valid
			std::copy(preprocessedImage.begin<float>(), preprocessedImage.end<float>(), src.begin());
		}
		if (ratioInputIndex < inputTensorValues.size() && !inputTensorValues[ratioInputIndex].empty()) {
			inputTensorValues[ratioInputIndex][0] = downsampleRatio;
		}
	}

	virtual void runNetworkInference(const std::unique_ptr<Ort::Session> &session,
					 const std::vector<Ort::AllocatedStringPtr> &inputNames,
					 const std::vector<Ort::AllocatedStringPtr> &outputNames,
					 const std::vector<Ort::Value> &inputTensor, std::vector<Ort::Value> &outputTensor)
	{
		UNUSED_PARAMETER(outputTensor);
		if (inputNames.empty() || outputNames.empty() || inputTensor.size() != inputNames.size()) {
			obs_log(LOG_INFO, "Skip network inference. Inputs or outputs are null.");
			return;
		}

		Ort::MemoryInfo memoryInfo =
			Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtDeviceAllocator, OrtMemType::OrtMemTypeDefault);

		std::vector<const char *> rawInputNames;
		std::vector<Ort::Value> inputs;
		for (size_t i = 0; i < inputNames.size(); i++) {
			const char *name = inputNames[i].get();
			rawInputNames.push_back(name);

			const int r = recurrentIndexFromName(name, 'i');
			if (r >= 0 && recurrent.size() == 4) {
				inputs.push_back(std::move(recurrent[(size_t)r]));
				continue;
			}
			// Non-owning view over the preallocated buffer
			auto info = inputTensor[i].GetTensorTypeAndShapeInfo();
			std::vector<int64_t> shape = info.GetShape();
			float *data = const_cast<float *>(inputTensor[i].GetTensorData<float>());
			inputs.push_back(Ort::Value::CreateTensor<float>(memoryInfo, data, info.GetElementCount(),
									 shape.data(), shape.size()));
		}

		std::vector<const char *> rawOutputNames;
		for (auto &outputName : outputNames) {
			rawOutputNames.push_back(outputName.get());
		}

		std::vector<Ort::Value> outputs = session->Run(Ort::RunOptions{nullptr}, rawInputNames.data(),
							       inputs.data(), inputs.size(), rawOutputNames.data(),
							       rawOutputNames.size());

		std::vector<Ort::Value> newRecurrent(4);
		size_t recurrentFound = 0;
		for (size_t i = 0; i < outputs.size(); i++) {
			const char *name = rawOutputNames[i];
			const int r = recurrentIndexFromName(name, 'o');
			if (r >= 0) {
				newRecurrent[(size_t)r] = std::move(outputs[i]);
				recurrentFound++;
			} else if (std::strcmp(name, "pha") == 0) {
				lastAlpha = std::move(outputs[i]);
				hasAlpha = true;
			}
		}
		if (recurrentFound == 4) {
			recurrent = std::move(newRecurrent);
		} else {
			recurrent.clear();
		}
	}

	virtual cv::Mat getNetworkOutput(const std::vector<std::vector<int64_t>> &outputDims,
					 std::vector<std::vector<float>> &outputTensorValues)
	{
		if (!hasAlpha) {
			return ModelBCHW::getNetworkOutput(outputDims, outputTensorValues);
		}
		std::vector<int64_t> shape = lastAlpha.GetTensorTypeAndShapeInfo().GetShape();
		// pha is (1, 1, H, W)
		const int h = (int)shape[2];
		const int w = (int)shape[3];
		return cv::Mat(h, w, CV_32FC1, lastAlpha.GetTensorMutableData<float>());
	}

	virtual void assignOutputToInput(std::vector<std::vector<float>> &, std::vector<std::vector<float>> &)
	{
		// recurrent state is carried in runNetworkInference()
	}
};

#endif /* MODELRVM_H */
