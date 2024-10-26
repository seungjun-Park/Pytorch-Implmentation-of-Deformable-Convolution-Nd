#pragma once

#include <torch/extension.h>
#include <ATen/div_rtn.h>
#include <ATen/native/utils/ParamUtils.h>

void check_deform_conv_backend(
	const at::Tensor& input,
	const at::Tensor& weight,
	const at::Tensor& offset_field,
	const at::Tensor& attn_mask,
	const at::Tensor& bias,
	const at::Tensor& grad_output,
	at::Backend location);

template<int64_t dim>
void check_deform_conv_shape(
	const at::Tensor& input,
	const at::Tensor& weight,
	const at::Tensor& offset_field,
	const at::Tensor& attn_mask,
	const at::Tensor& grad_output,
	at::IntArrayRef kernel_size,
	at::IntArrayRef stride,
	at::IntArrayRef padding,
	at::IntArrayRef dilation,
	const int64_t groups,
	const int64_t offset_field_channels_per_groups,
	const at::Tensor& bias
)
{
	auto input_shape = input.sizes();
	auto weight_shape = weight.sizes();
	auto offset_field_shape = offset_field.sizes();
	auto attn_mask_shape = attn_mask.sizes();

	TORCH_CHECK(input.dim() == 2 + dim);
	TORCH_CHECK(weight.dim() == 2 + dim);
	TORCH_CHECK(offset_field.dim() == 3 + dim);
	TORCH_CHECK(attn_mask.dim() == 3 + dim);

	TORCH_CHECK(kernel_size.size() == dim);
	TORCH_CHECK(stride.size() == dim);
	TORCH_CHECK(padding.size() == dim);
	TORCH_CHECK(dilation.size() == dim);

	TORCH_CHECK(groups > 0 && input_shape[1] % groups == 0);
	TORCH_CHECK(offset_field_channels_per_groups * groups <= input_shape[1]);

	TORCH_CHECK(weight_shape[1] == input_shape[1] / groups);

	if (bias.defined())
	{
		TORCH_CHECK(bias.sizes()[0] == weight_shape[0]);
	}
}

template <int64_t dim>
std::vector<int64_t> get_output_size(
	const at::Tensor& input,
	at::IntArrayRef kernel_size,
	at::IntArrayRef stride_size,
	at::IntArrayRef pad_size,
	at::IntArrayRef dilation_size);

template <int64_t dim>
std::vector<int64_t> get_output_size(
	const at::Tensor& input,
	const at::Tensor& weight,
	at::IntArrayRef kernel_size,
	at::IntArrayRef stride_size,
	at::IntArrayRef pad_size,
	at::IntArrayRef dilation_size);

template <int64_t dim>
std::vector<int64_t> get_output_size(
	const at::Tensor& input,
	at::IntArrayRef kernel_size,
	at::IntArrayRef stride_size,
	at::IntArrayRef pad_size,
	at::IntArrayRef dilation_size) {
	std::vector<int64_t> sizes;
	for (const auto index : c10::irange(dim)) {
		sizes.push_back(
			div_rtn<int64_t>(
				input.size(index + input.dim() - dim) + 2 * pad_size[index] -
				(dilation_size[index] * (kernel_size[index] - 1) + 1),
				stride_size[index]) +
			1);
	}
	return sizes;
}

template <int64_t dim>
std::vector<int64_t> get_output_size(
	const at::Tensor& input,
	const at::Tensor& weight,
	at::IntArrayRef kernel_size,
	at::IntArrayRef stride_size,
	at::IntArrayRef pad_size,
	at::IntArrayRef dilation_size) {
	auto output_size = get_output_size<dim>(
		input, kernel_size, stride_size, pad_size, dilation_size);
	output_size.insert(output_size.begin(), weight.size(0));
	if (input.dim() == dim + 2) {
		output_size.insert(output_size.begin(), input.size(0));
	}
	return output_size;
}
