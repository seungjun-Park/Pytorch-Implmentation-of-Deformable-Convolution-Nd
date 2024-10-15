#include <deform_conv.h>
#include <cpu/deform_conv_cpu.h>
#include <deform_conv_utils.h>
#include <utils.h>

#include <ATen/native/utils/ParamUtils.h>

#ifdef WITH_CUDA
#include <cuda/deform_conv_cuda.h>
#endif

at::Tensor deform_conv_forward(
	const at::Tensor& input,
	const at::Tensor& weight,
	const at::Tensor& offset_field,
	const at::Tensor& attn_mask,
	at::IntArrayRef kernel_size,
	at::IntArrayRef stride,
	at::IntArrayRef padding,
	at::IntArrayRef dilation,
	int64_t groups,
	const c10::optional<at::Tensor>& bias_opt
)
{
	c10::MaybeOwned<at::Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias_opt);
	const at::Tensor& bias = *bias_maybe_owned;

	auto k = weight.dim();
	int64_t dim = k - 2;

	bool is_batched = input.dim() == dim + 2;
	const at::Tensor batched_input = is_batched ? input.contiguous() : input.contiguous().unsqueeze(0);

	TORCH_CHECK(dim > 0, "weight should have at least three dimensions");
	TORCH_CHECK(groups > 0, "non-positive groups is not supported");

	at::Tensor undefined;

	at::Backend backend = input.is_cuda() ? at::Backend::CUDA : at::Backend::CPU;

	check_deform_conv_backend(
		input,
		weight,
		offset_field,
		attn_mask,
		bias,
		undefined,
		backend
	);

	// the function pointer for each backend implementation
	at::Tensor(*deform_conv_forward_func)(
		const at::Tensor& input,
		const at::Tensor& weight,
		const at::Tensor& offset_field,
		const at::Tensor& attn_mask,
		at::IntArrayRef kernel_size,
		at::IntArrayRef stride,
		at::IntArrayRef padding,
		at::IntArrayRef dilation,
		int64_t groups,
		const at::Tensor& bias) = nullptr;

	if (input.is_cuda())
	{
#ifdef WITH_CUDA
		deform_conv_forward_func = deform_conv_nd_forward_cuda;
#endif // WITH_CUDA
	}
	else
	{
		deform_conv_forward_func = deform_conv_nd_forward_cpu;
	}

	TORCH_CHECK(deform_conv_forward_func != nullptr);

	at::Tensor output = (*deform_conv_forward_func)(
		batched_input,
		weight,
		offset_field,
		attn_mask,
		at::native::expand_param_if_needed(kernel_size, "kernel_size", dim),
		at::native::expand_param_if_needed(stride, "stride", dim),
		at::native::expand_param_if_needed(padding, "padding", dim),
		at::native::expand_param_if_needed(dilation, "dilation", dim),
		groups,
		bias
	);

	return is_batched ? std::move(output) : output.squeeze(0);
}

std::vector<at::Tensor> deform_conv_backward(
	const at::Tensor& input,
	const at::Tensor& weight,
	const at::Tensor& offset_field,
	const at::Tensor& attn_mask,
	const at::Tensor& grad_output,
	at::IntArrayRef kernel_size,
	at::IntArrayRef stride,
	at::IntArrayRef padding,
	at::IntArrayRef dilation,
	int64_t groups,
	const c10::optional<at::Tensor>& bias_opt
)
{
	c10::MaybeOwned<at::Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias_opt);
	const at::Tensor& bias = *bias_maybe_owned;

	auto k = weight.dim();
	int64_t dim = k - 2;

	bool is_batched = input.dim() == dim + 2;
	const at::Tensor batched_input = is_batched ? input.contiguous() : input.contiguous().unsqueeze(0);

	TORCH_CHECK(dim > 0, "weight should have at least three dimensions");
	TORCH_CHECK(groups > 0, "non-positive groups is not supported");

	at::Backend backend = input.is_cuda() ? at::Backend::CUDA : at::Backend::CPU;

	check_deform_conv_backend(
		input,
		weight,
		offset_field,
		attn_mask,
		bias,
		grad_output,
		backend
	);

	// the function pointer for each backend implementation
	std::vector<at::Tensor> (*deform_conv_backward_func)(
		const at::Tensor&,
		const at::Tensor&,
		const at::Tensor&,
		const at::Tensor&,
		const at::Tensor&,
		at::IntArrayRef,
		at::IntArrayRef,
		at::IntArrayRef,
		at::IntArrayRef,
		int64_t,
		const at::Tensor&) = nullptr;

	if (input.is_cuda())
	{
#ifdef WITH_CUDA
		deform_conv_backward_func = deform_conv_nd_backward_cuda;
#endif // WITH_CUDA
	}
	else
	{
		deform_conv_backward_func = deform_conv_nd_backward_cpu;
	}

	std::vector<at::Tensor> grads = deform_conv_backward_func(
		batched_input,
		weight,
		offset_field,
		attn_mask,
		grad_output,
		at::native::expand_param_if_needed(kernel_size, "kernel_size", dim),
		at::native::expand_param_if_needed(stride, "stride", dim),
		at::native::expand_param_if_needed(padding, "padding", dim),
		at::native::expand_param_if_needed(dilation, "dilation", dim),
		groups,
		bias
	);

	if (!is_batched)
	{
		grads[0] = grads[0].squeeze(0);
		grads[2] = grads[2].squeeze(0);
		grads[3] = grads[3].squeeze(0);
	}

	return grads;
}