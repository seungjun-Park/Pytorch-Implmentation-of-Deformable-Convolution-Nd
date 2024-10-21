#pragma once
#include <torch/extension.h>

#include <vector>

at::Tensor deform_conv_nd_forward_cpu(
	const at::Tensor& input,
	const at::Tensor& weight,
	const at::Tensor& offset_field,
	const at::Tensor& attn_mask,
	at::IntArrayRef kernel_size,
	at::IntArrayRef stride,
	at::IntArrayRef padding,
	at::IntArrayRef dilation,
	const int32_t groups,
	const int32_t offset_field_channels_per_groups,
	const at::Tensor& bias);

std::vector<at::Tensor> deform_conv_nd_backward_cpu(
	const at::Tensor& input,
	const at::Tensor& weight,
	const at::Tensor& offset_field,
	const at::Tensor& attn_mask,
	const at::Tensor& grad_output,
	at::IntArrayRef kernel_size,
	at::IntArrayRef stride,
	at::IntArrayRef padding,
	at::IntArrayRef dilation,
	const int32_t groups,
	const int32_t offset_field_channels_per_groups,
	const at::Tensor& bias);