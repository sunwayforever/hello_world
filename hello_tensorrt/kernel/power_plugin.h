// 2022-06-14 10:53
#include <iostream>
#include <numeric>
#include <string>

#include "NvCaffeParser.h"
#include "my_plugin.h"

extern void Power(float*, const float*, float, float, float, int, cudaStream_t);

using namespace nvinfer1;

class PowerPlugin : public MyPlugin {
   public:
    PowerPlugin() {}
    PowerPlugin(const PluginFieldCollection fc)
        : mPower(0.0), mScale(1.0), mShift(0.0) {
        for (int i = 0; i < fc.nbFields; i++) {
            auto field = fc.fields[i];
            if (std::string(field.name) == "power") {
                this->mPower = *((float*)field.data);
            }
            if (std::string(field.name) == "scale") {
                this->mScale = *((float*)field.data);
            }
            if (std::string(field.name) == "shift") {
                this->mShift = *((float*)field.data);
            }
        }
    }

    PowerPlugin(const void* data, size_t length) {
        mPower = ((float*)data)[0];
        mShift = ((float*)data)[1];
        mScale = ((float*)data)[2];
        mInputSize = ((int*)data)[3];
    }

   public:
    int getNbOutputs() const noexcept override { return 1; }

    Dims getOutputDimensions(
        int index, const Dims* inputs, int nbInputDims) noexcept override {
        return *inputs;
    }

    size_t getWorkspaceSize(int maxBatchSize) const noexcept override {
        return 0;
    }

    int enqueue(
        int batchSize, const void* const* inputs, void* const* outputs,
        void* workspace, cudaStream_t stream) noexcept override {
        float* dst = reinterpret_cast<float*>(outputs[0]);
        const float* src = reinterpret_cast<const float*>(inputs[0]);
        Power(dst, src, mScale, mPower, mShift, mInputSize, stream);
        return 0;
    }

    size_t getSerializationSize() const noexcept override { return 16; }
    void serialize(void* buffer) const noexcept override {
        ((float*)buffer)[0] = mPower;
        ((float*)buffer)[1] = mShift;
        ((float*)buffer)[2] = mScale;
        ((int*)buffer)[3] = mInputSize;
    }
    void configurePlugin(
        const PluginTensorDesc* in, int nbInput, const PluginTensorDesc* out,
        int nbOutput) noexcept override {
        auto dims = in[0].dims;
        mInputSize = std::accumulate(
            dims.d, dims.d + dims.nbDims, 1, std::multiplies<int>());
    }

    bool supportsFormatCombination(
        int pos, const PluginTensorDesc* inOut, int nbInputs,
        int nbOutputs) const noexcept override {
        return inOut[pos].type == DataType::kFLOAT &&
               inOut[pos].format == inOut[0].format;
    }

    const char* getPluginType() const noexcept override { return "POWER"; }

    IPluginV2Ext* clone() const noexcept override {
        auto* plugin = new PowerPlugin(*this);
        return plugin;
    }

   private:
    int mInputSize;
    float mPower;
    float mShift;
    float mScale;
};
