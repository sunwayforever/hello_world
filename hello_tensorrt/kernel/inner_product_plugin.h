// 2022-06-14 10:53
#include <cstring>
#include <iostream>
#include <numeric>
#include <string>

#include "NvCaffeParser.h"
#include "my_plugin.h"

extern void InnerProduct(
    float*, const float*, int, int, float*, float*, cudaStream_t);

using namespace nvinfer1;

class InnerProductPlugin : public MyPlugin {
   public:
    InnerProductPlugin(const PluginFieldCollection fc) {
        for (int i = 0; i < fc.nbFields; i++) {
            auto field = fc.fields[i];
            if (std::string(field.name) == "num_output") {
                this->mOutputSize = *((int*)field.data);
            }
            if (std::string(field.name) == "kernel_weights") {
                this->mKernelWeights = *(Weights*)field.data;
            }
            if (std::string(field.name) == "bias_weights") {
                this->mBiasWeights = *(Weights*)field.data;
            }
        }
    }

    InnerProductPlugin(const void* data, size_t length) {
        mInputSize = ((int*)data)[0];
        mOutputSize = ((int*)data)[1];
        int kc = ((int*)data)[2];
        int bc = ((int*)data)[3];

        float* kernel = (float*)malloc(kc * 4);
        float* bias = (float*)malloc(bc * 4);

        memcpy(kernel, ((int*)data) + 4, kc * 4);
        memcpy(bias, ((int*)data) + 4 + kc, bc * 4);

        mKernelWeights = Weights{
            .type = DataType::kFLOAT,
            .values = kernel,
            .count = kc,
        };

        mBiasWeights = Weights{
            .type = DataType::kFLOAT,
            .values = bias,
            .count = bc,
        };
    }

   public:
    int getNbOutputs() const noexcept override { return 1; }

    Dims getOutputDimensions(
        int index, const Dims* inputs, int nbInputDims) noexcept override {
        return Dims{1, {mOutputSize}};
    }

    size_t getWorkspaceSize(int maxBatchSize) const noexcept override {
        return 0;
    }

    int enqueue(
        int batchSize, const void* const* inputs, void* const* outputs,
        void* workspace, cudaStream_t stream) noexcept override {
        float* dst = reinterpret_cast<float*>(outputs[0]);
        const float* src = reinterpret_cast<const float*>(inputs[0]);
        InnerProduct(
            dst, src, mInputSize, mOutputSize, (float*)mKernelWeights.values,
            (float*)mBiasWeights.values, stream);
        return 0;
    }

    size_t getSerializationSize() const noexcept override {
        return (4 + mKernelWeights.count + mBiasWeights.count) * 4;
    }

    void serialize(void* buffer) const noexcept override {
        ((int*)buffer)[0] = mInputSize;
        ((int*)buffer)[1] = mOutputSize;
        ((int*)buffer)[2] = mKernelWeights.count;
        ((int*)buffer)[3] = mBiasWeights.count;
        memcpy(
            ((int*)buffer) + 4, mKernelWeights.values,
            mKernelWeights.count * 4);
        memcpy(
            ((int*)buffer) + 4 + mKernelWeights.count, mBiasWeights.values,
            mBiasWeights.count * 4);
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
        return inOut[pos].format == TensorFormat::kLINEAR &&
               inOut[pos].type == DataType::kFLOAT;
    }

    const char* getPluginType() const noexcept override {
        return "INNER_PRODUCT";
    }

    IPluginV2Ext* clone() const noexcept override {
        auto* plugin = new InnerProductPlugin(*this);
        return plugin;
    }

   private:
    int mOutputSize;
    int mInputSize;
    Weights mKernelWeights;
    Weights mBiasWeights;
};
