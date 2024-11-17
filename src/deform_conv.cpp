#include <torch/extension.h>
#include <deform_conv_utils.h>
#include <array_utils.h>
#include <type_utils.h>

#include <ATen/native/utils/ParamUtils.h>
#include <ATen/autocast_mode.h>

template<int8_t dim>
at::Tensor deform_conv_forward(
	const at::Tensor& input,
	const at::Tensor& weight,
	const at::Tensor& offset_field,
	const at::Tensor& attn_mask,
	at::IntArrayRef kernel_size,
	at::IntArrayRef stride,
	at::IntArrayRef padding,
	at::IntArrayRef dilation,
	const int64_t groups,
	const int64_t deformable_groups_per_groups,
	const double_t offset_scale,
	const bool fix_center,
	const at::Tensor& bias
)
{
	bool is_channels_last = check_is_channels_last<dim>(input);

	bool is_batched = input.dim() == dim + 2;
	if (!is_batched)
	{
		make_batched_tensor<dim>(const_cast<at::Tensor&>(input), is_channels_last);
		make_batched_tensor<dim>(const_cast<at::Tensor&>(offset_field), is_channels_last);
		make_batched_tensor<dim>(const_cast<at::Tensor&>(attn_mask), is_channels_last);
	}

	TORCH_CHECK(dim > 0, "weight should have at least three dimensions");
	TORCH_CHECK(groups > 0, "non-positive groups is not supported");

	// the function is located torch::ops::custom_op namespace.
	std::string func_name = "custom_op::deform_conv";
	func_name += std::to_string(dim) + "d_forward";

	static auto op = torch::Dispatcher::singleton()
		.findSchemaOrThrow(func_name.c_str(), "")
		.typed<decltype(deform_conv_forward<dim>)>();
 
	at::Tensor output;

	if (!torch::GradMode::is_enabled())
	{
		torch::NoGradGuard no_grad;
		output = op.call(
			input,
			weight,
			offset_field,
			attn_mask,
			at::native::expand_param_if_needed(kernel_size, "kernel_size", dim),
			at::native::expand_param_if_needed(stride, "stride", dim),
			at::native::expand_param_if_needed(padding, "padding", dim),
			at::native::expand_param_if_needed(dilation, "dilation", dim),
			groups,
			deformable_groups_per_groups,
			offset_scale,
			fix_center,
			bias
		);
	}
	else
	{
		output = op.call(
			input,
			weight,
			offset_field,
			attn_mask,
			at::native::expand_param_if_needed(kernel_size, "kernel_size", dim),
			at::native::expand_param_if_needed(stride, "stride", dim),
			at::native::expand_param_if_needed(padding, "padding", dim),
			at::native::expand_param_if_needed(dilation, "dilation", dim),
			groups,
			deformable_groups_per_groups,
			offset_scale,
			fix_center,
			bias
		);
	}

	return (is_batched) ? output : output.squeeze(0);
}

template<int8_t dim>
torch::autograd::tensor_list deform_conv_backward(
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
	const int64_t deformable_groups_per_groups,
	const double_t offset_scale,
	const bool fix_center,
	const at::Tensor& bias
)
{
	bool is_channels_last = check_is_channels_last<dim>(input);

	bool is_batched = input.dim() == dim + 2;
	if (!is_batched)
	{
		make_batched_tensor<dim>(const_cast<at::Tensor&>(input), is_channels_last);
		make_batched_tensor<dim>(const_cast<at::Tensor&>(offset_field), is_channels_last);
		make_batched_tensor<dim>(const_cast<at::Tensor&>(attn_mask), is_channels_last);
		make_batched_tensor<dim>(const_cast<at::Tensor&>(grad_output), is_channels_last);
	}

	TORCH_CHECK(dim > 0, "weight should have at least three dimensions");
	TORCH_CHECK(groups > 0, "non-positive groups is not supported");

	// the function is located torch::ops::custom_op namespace.
	std::string func_name = "custom_op::deform_conv";
	func_name += std::to_string(dim) + "d_backward";

	static auto op = torch::Dispatcher::singleton()
		.findSchemaOrThrow(func_name.c_str(), "")
		.typed<decltype(deform_conv_backward<dim>)>();

	torch::autograd::tensor_list grads;

	if (!torch::GradMode::is_enabled())
	{
		torch::NoGradGuard no_grad;
		grads = std::move(op.call(
			input,
			weight,
			offset_field,
			attn_mask,
			grad_output,
			at::native::expand_param_if_needed(kernel_size, "kernel_size", dim),
			at::native::expand_param_if_needed(stride, "stride", dim),
			at::native::expand_param_if_needed(padding, "padding", dim),
			at::native::expand_param_if_needed(dilation, "dilation", dim),
			groups,
			deformable_groups_per_groups,
			offset_scale,
			fix_center,
			bias
		));
	}
	else
	{
		grads = std::move(op.call(
			input,
			weight,
			offset_field,
			attn_mask,
			grad_output,
			at::native::expand_param_if_needed(kernel_size, "kernel_size", dim),
			at::native::expand_param_if_needed(stride, "stride", dim),
			at::native::expand_param_if_needed(padding, "padding", dim),
			at::native::expand_param_if_needed(dilation, "dilation", dim),
			groups,
			deformable_groups_per_groups,
			offset_scale,
			fix_center,
			bias
		));
	}

	if (!is_batched)
	{
		grads[0] = grads[0].squeeze(0);
		grads[2] = grads[2].squeeze(0);
		grads[3] = grads[3].squeeze(0);
	}

	return grads;
}


template<int8_t dim>// To supoort autograd.
class DeformConvNdFunction : public torch::autograd::Function<DeformConvNdFunction<dim>>
{
public:
	static at::Tensor forward(
		torch::autograd::AutogradContext* ctx,
		const at::Tensor& input,
		const at::Tensor& weight,
		const at::Tensor& offset_field,
		const at::Tensor& attn_mask,
		at::IntArrayRef kernel_size,
		at::IntArrayRef stride,
		at::IntArrayRef padding,
		at::IntArrayRef dilation,
		const int64_t groups,
		const int64_t deformable_groups_per_groups,
		const double_t offset_scale,
		const bool fix_center,
		const at::Tensor& bias
	) {
		// at::AutoNonVariableTypeMode g; will deprecated 1.10 version.
		at::AutoDispatchBelowADInplaceOrView g;

		ctx->save_for_backward(
			{ input, weight, offset_field, attn_mask, bias }
		);

		ctx->saved_data["kernel_size"] = kernel_size;
		ctx->saved_data["stride"] = stride;
		ctx->saved_data["padding"] = padding;
		ctx->saved_data["dilation"] = dilation;
		ctx->saved_data["groups"] = groups;
		ctx->saved_data["deformable_groups_per_groups"] = deformable_groups_per_groups;
		ctx->saved_data["offset_scale"] = offset_scale;
		ctx->saved_data["fix_center"] = fix_center;

		return deform_conv_forward<dim>(
			input,
			weight,
			offset_field,
			attn_mask,
			kernel_size,
			stride,
			padding,
			dilation,
			groups,
			deformable_groups_per_groups,
			offset_scale,
			fix_center,
			bias
		);
	}

	static torch::autograd::variable_list backward(
		torch::autograd::AutogradContext* ctx,
		torch::autograd::tensor_list grad_outputs
	) {
		at::Tensor grad_output = grad_outputs[0];
		torch::autograd::tensor_list tensors = ctx->get_saved_variables();

		return deform_conv_backward<dim>(
			tensors[0],
			tensors[1],
			tensors[2],
			tensors[3],
			grad_output,
			ctx->saved_data["kernel_size"].toIntVector(),
			ctx->saved_data["stride"].toIntVector(),
			ctx->saved_data["padding"].toIntVector(),
			ctx->saved_data["dilation"].toIntVector(),
			ctx->saved_data["groups"].toInt(),
			ctx->saved_data["deformable_groups_per_groups"].toInt(),
			ctx->saved_data["offset_scale"].toDouble(),
			ctx->saved_data["fix_center"].toBool(),
			tensors[4]
		);
	}
};

template<int8_t dim>
at::Tensor deform_conv_nd_autograd(
	const at::Tensor& input,
	const at::Tensor& weight,
	const at::Tensor& offset_field,
	const at::Tensor& attn_mask,
	at::IntArrayRef kernel_size,
	at::IntArrayRef stride,
	at::IntArrayRef padding,
	at::IntArrayRef dilation,
	const int64_t groups,
	const int64_t deformable_groups_per_groups,
	const double_t offset_scale,
	const bool fix_center,
	const at::Tensor& bias)
{
	return DeformConvNdFunction<dim>::apply(
		input,
		weight,
		offset_field,
		attn_mask,
		kernel_size,
		stride,
		padding,
		dilation,
		groups,
		deformable_groups_per_groups,
		offset_scale,
		fix_center,
		bias
	);
}

template<int8_t dim>
at::Tensor deform_conv_nd_autocast_cpu(
	const at::Tensor& input,
	const at::Tensor& weight,
	const at::Tensor& offset_field,
	const at::Tensor& attn_mask,
	at::IntArrayRef kernel_size,
	at::IntArrayRef stride,
	at::IntArrayRef padding,
	at::IntArrayRef dilation,
	const int64_t groups,
	const int64_t deformable_groups_per_groups,
	const double_t offset_scale,
	const bool fix_center,
	const at::Tensor& bias)
{
	c10::impl::ExcludeDispatchKeyGuard no_autocast(c10::DispatchKey::Autocast);
	c10::DeviceType device = input.device().type();
	c10::ScalarType dtype = at::autocast::get_autocast_cpu_dtype();
	// For cuda 12.4 above with pytorch 2.4.0
	// c10::ScalarType dtype = at::autocast::get_autocast_dtype(at::kCPU);
	return deform_conv_nd_autograd<dim>(
		at::autocast::cached_cast(dtype, input, device),
		at::autocast::cached_cast(dtype, weight, device),
		at::autocast::cached_cast(dtype, offset_field, device),
		at::autocast::cached_cast(dtype, attn_mask, device),
		kernel_size,
		stride,
		padding,
		dilation,
		groups,
		deformable_groups_per_groups,
		offset_scale,
		fix_center,
		at::autocast::cached_cast(dtype, bias, device)
		);
}

template<int8_t dim>
at::Tensor deform_conv_nd_autocast_cuda(
	const at::Tensor& input,
	const at::Tensor& weight,
	const at::Tensor& offset_field,
	const at::Tensor& attn_mask,
	at::IntArrayRef kernel_size,
	at::IntArrayRef stride,
	at::IntArrayRef padding,
	at::IntArrayRef dilation,
	const int64_t groups,
	const int64_t deformable_groups_per_groups,
	const double_t offset_scale,
	const bool fix_center,
	const at::Tensor& bias)
{
	c10::impl::ExcludeDispatchKeyGuard no_autocast(c10::DispatchKey::Autocast);
	c10::DeviceType device = input.device().type();
	c10::ScalarType dtype = at::autocast::get_autocast_gpu_dtype();
	// For cuda 12.4 above with pytorch 2.4.0
	// c10::ScalarType dtype = at::autocast::get_autocast_dtype(at::kCUDA);
	return deform_conv_nd_autograd<dim>(
		at::autocast::cached_cast(dtype, input, device),
		at::autocast::cached_cast(dtype, weight, device),
		at::autocast::cached_cast(dtype, offset_field, device),
		at::autocast::cached_cast(dtype, attn_mask, device),
		kernel_size,
		stride,
		padding,
		dilation,
		groups,
		deformable_groups_per_groups,
		offset_scale,
		fix_center,
		at::autocast::cached_cast(dtype, bias, device)
	);
}

TORCH_LIBRARY(custom_op, m)
{
	m.def("deform_conv1d(Tensor input, Tensor weight, Tensor offset_field, Tensor attn_mask, int[] kernel_size, int[] stride, int[] padding, int[] dilation, int groups, int deformable_groups_per_groups, float offset_scale, bool fix_center, Tensor bias) -> Tensor");
	m.def("deform_conv2d(Tensor input, Tensor weight, Tensor offset_field, Tensor attn_mask, int[] kernel_size, int[] stride, int[] padding, int[] dilation, int groups, int deformable_groups_per_groups, float offset_scale, bool fix_center, Tensor bias) -> Tensor");
	m.def("deform_conv3d(Tensor input, Tensor weight, Tensor offset_field, Tensor attn_mask, int[] kernel_size, int[] stride, int[] padding, int[] dilation, int groups, int deformable_groups_per_groups, float offset_scale, bool fix_center, Tensor bias) -> Tensor");

	m.def("deform_conv1d_forward(Tensor input, Tensor weight, Tensor offset_field, Tensor attn_mask, int[] kernel_size, int[] stride, int[] padding, int[] dilation, int groups, int deformable_groups_per_groups, float offset_scale, bool fix_center, Tensor bias) -> Tensor");
	m.def("deform_conv2d_forward(Tensor input, Tensor weight, Tensor offset_field, Tensor attn_mask, int[] kernel_size, int[] stride, int[] padding, int[] dilation, int groups, int deformable_groups_per_groups, float offset_scale, bool fix_center, Tensor bias) -> Tensor");
	m.def("deform_conv3d_forward(Tensor input, Tensor weight, Tensor offset_field, Tensor attn_mask, int[] kernel_size, int[] stride, int[] padding, int[] dilation, int groups, int deformable_groups_per_groups, float offset_scale, bool fix_center, Tensor bias) -> Tensor");

	m.def("deform_conv1d_backward(Tensor input, Tensor weight, Tensor offset_field, Tensor attn_mask, Tensor grad_output, int[] kernel_size, int[] stride, int[] padding, int[] dilation, int groups, int deformable_groups_per_groups, float offset_scale, bool fix_center, Tensor bias) -> Tensor[]");
	m.def("deform_conv2d_backward(Tensor input, Tensor weight, Tensor offset_field, Tensor attn_mask, Tensor grad_output, int[] kernel_size, int[] stride, int[] padding, int[] dilation, int groups, int deformable_groups_per_groups, float offset_scale, bool fix_center, Tensor bias) -> Tensor[]");
	m.def("deform_conv3d_backward(Tensor input, Tensor weight, Tensor offset_field, Tensor attn_mask, Tensor grad_output, int[] kernel_size, int[] stride, int[] padding, int[] dilation, int groups, int deformable_groups_per_groups, float offset_scale, bool fix_center, Tensor bias) -> Tensor[]");
}

TORCH_LIBRARY_IMPL(custom_op, Autograd, m)
{
	m.impl("deform_conv1d", &deform_conv_nd_autograd<1>);
	m.impl("deform_conv2d", &deform_conv_nd_autograd<2>);
	m.impl("deform_conv3d", &deform_conv_nd_autograd<3>);
}

TORCH_LIBRARY_IMPL(custom_op, AutocastCPU, m)
{
	m.impl("deform_conv1d", &deform_conv_nd_autocast_cpu<1>);
	m.impl("deform_conv2d", &deform_conv_nd_autocast_cpu<2>);
	m.impl("deform_conv3d", &deform_conv_nd_autocast_cpu<3>);
}

TORCH_LIBRARY_IMPL(custom_op, AutocastCUDA, m)
{
	m.impl("deform_conv1d", &deform_conv_nd_autocast_cuda<1>);
	m.impl("deform_conv2d", &deform_conv_nd_autocast_cuda<2>);
	m.impl("deform_conv3d", &deform_conv_nd_autocast_cuda<3>);
}


// Dummy to prevent python link error.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
	m.doc() = "Pytorch implementation of deformable convolution Nd";
}

#include <torch/extension.h>
#include <iostream>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <chrono>

#include <GPUInfo.h>
#include <random>
#include <vector>
#include <cuda_fp16.h>
#include <type_traits>
#include <array_utils.h>
#include <type_utils.h>
#include <ATen/Dispatch.h>

using scalar_t = float;
at::ScalarType dtype = at::ScalarType::Float;

std::vector<scalar_t> test_backward()
{
	constexpr int dim = 2;

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<int64_t> batch_gen(1, 4);
	std::uniform_int_distribution<int64_t> groups_gen(1, 4);
	std::uniform_int_distribution<int64_t> grouped_in_gen(8, 16);
	std::uniform_int_distribution<int64_t> grouped_out_gen(8, 16);
	std::uniform_int_distribution<int64_t> deformable_g_gen(2, 3);
	std::uniform_int_distribution<int64_t> size_gen(32, 128);

	int64_t batch_size = batch_gen(gen);
	int64_t groups = groups_gen(gen);
	int64_t deformable_groups = deformable_g_gen(gen);
	int64_t grouped_in_channels = grouped_in_gen(gen) * deformable_groups;
	int64_t grouped_out_channels = grouped_out_gen(gen);
	int64_t in_channels = groups * grouped_in_channels;
	int64_t out_channels = groups * grouped_out_channels;

	int64_t size = size_gen(gen);
	int64_t k = 3;
	int64_t s = 1;
	int64_t p = 1;
	int64_t d = 1;

	at::IntArrayRef input_size({ batch_size, in_channels, size, size });
	at::IntArrayRef output_size({ batch_size, out_channels, size, size });
	at::IntArrayRef kernel_size({ k, k });
	at::IntArrayRef stride({ s, s });
	at::IntArrayRef padding({ p , p });
	at::IntArrayRef dilation({ d, d });

	int64_t kernel_sizes = c10::multiply_integers(kernel_size);
	int64_t input_sizes = c10::multiply_integers(input_size.slice(2));
	int64_t output_sizes = c10::multiply_integers(output_size.slice(2));

	double offset_scale = 1.0;
	bool fix_center = true;

	at::Tensor input = at::randn({ batch_size, in_channels, size, size }, torch::kCPU).to(dtype);
	at::Tensor weight = at::randn({ out_channels, grouped_in_channels, k, k }, torch::kCPU).to(dtype);
	at::Tensor bias = at::randn({ out_channels }, torch::kCPU).to(dtype);
	at::Tensor offset_field = at::randn({ batch_size, groups * deformable_groups * (kernel_sizes - (int)fix_center) * dim, size, size }, torch::kCPU).to(dtype);
	at::Tensor attn_mask = at::randn({ batch_size, groups * deformable_groups * kernel_sizes, size, size }, torch::kCPU).to(dtype);

	at::Tensor grad_output = at::randn({ batch_size, out_channels, size, size }, torch::kCPU).to(dtype);

	auto grads_cpu = deform_conv_backward<dim>(
		input,
		weight,
		offset_field,
		attn_mask,
		grad_output,
		k,
		s,
		p,
		d,
		groups,
		deformable_groups,
		offset_scale,
		fix_center,
		bias
	);

	auto grads_cuda = deform_conv_backward<dim>(
		input.cuda(),
		weight.cuda(),
		offset_field.cuda(),
		attn_mask.cuda(),
		grad_output.cuda(),
		k,
		s,
		p,
		d,
		groups,
		deformable_groups,
		offset_scale,
		fix_center,
		bias.cuda()
	);


	std::vector<scalar_t> diffs;

	for (size_t i = 0; i < grads_cpu.size(); i++)
	{
		if (!grads_cpu[i].defined())
		{
			continue;
		}

		auto diff = torch::abs(grads_cpu[i] - grads_cuda[i].cpu()).mean();
		diffs.push_back(*diff.const_data_ptr<scalar_t>());
	}

	return diffs;
}


float test_forward()
{
	constexpr int dim = 2;

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<int64_t> batch_gen(1, 4);
	std::uniform_int_distribution<int64_t> groups_gen(1, 4);
	std::uniform_int_distribution<int64_t> grouped_in_gen(8, 16);
	std::uniform_int_distribution<int64_t> grouped_out_gen(8, 16);
	std::uniform_int_distribution<int64_t> deformable_g_gen(2, 3);
	std::uniform_int_distribution<int64_t> size_gen(32, 128);

	int64_t batch_size = batch_gen(gen);
	int64_t groups = groups_gen(gen);
	int64_t deformable_groups = deformable_g_gen(gen);
	int64_t grouped_in_channels = grouped_in_gen(gen) * deformable_groups;
	int64_t grouped_out_channels = grouped_out_gen(gen);
	int64_t in_channels = groups * grouped_in_channels;
	int64_t out_channels = groups * grouped_out_channels;

	int64_t size = size_gen(gen);
	int64_t k = 3;
	int64_t s = 1;
	int64_t p = 1;
	int64_t d = 1;

	at::IntArrayRef input_size({ batch_size, in_channels, size, size });
	at::IntArrayRef output_size({ batch_size, out_channels, size, size });
	at::IntArrayRef kernel_size({ k, k });
	at::IntArrayRef stride({ s, s });
	at::IntArrayRef padding({ p , p });
	at::IntArrayRef dilation({ d, d });

	int64_t kernel_sizes = c10::multiply_integers(kernel_size);
	int64_t input_sizes = c10::multiply_integers(input_size.slice(2));
	int64_t output_sizes = c10::multiply_integers(output_size.slice(2));

	double_t offset_scale = 0.1;
	bool fix_center = true;

	at::Tensor input = at::randn({ batch_size, in_channels, size, size }, torch::kCPU).to(dtype).to(at::MemoryFormat::ChannelsLast);
	at::Tensor weight = at::randn({ out_channels, grouped_in_channels, k, k }, torch::kCPU).to(dtype);
	at::Tensor bias = at::randn({ out_channels }, torch::kCPU).to(dtype);
	at::Tensor offset_field = at::randn({ batch_size, groups * deformable_groups * (kernel_sizes - (int)fix_center) * dim, size, size }, torch::kCPU).to(dtype).to(at::MemoryFormat::ChannelsLast);
	at::Tensor attn_mask = at::randn({ batch_size, groups * deformable_groups * kernel_sizes , size, size }, torch::kCPU).to(dtype).to(at::MemoryFormat::ChannelsLast);

	at::Tensor output_cpu = deform_conv_nd_autograd<dim>(
		input,
		weight,
		offset_field,
		attn_mask,
		k,
		s,
		p,
		d,
		groups,
		deformable_groups,
		offset_scale,
		fix_center,
		bias
	);

	at::Tensor output_cuda = deform_conv_nd_autograd<dim>(
		input.cuda(),
		weight.cuda(),
		offset_field.cuda(),
		attn_mask.cuda(),
		k,
		s,
		p,
		d,
		groups,
		deformable_groups,
		offset_scale,
		fix_center,
		bias.cuda()
	);

	auto diff = torch::abs(output_cpu - output_cuda.cpu()).mean();

	return *diff.const_data_ptr<scalar_t>();
}


int main()
{
	int64_t test_iter = 100;


	double avg_diff = 0.f;

	/*try
	{
		for (size_t i = 0; i < test_iter; i++)
		{
			float diff = test_forward();
			avg_diff += diff;
			std::cout << "avg_diff[" << i << "]: " << avg_diff / (1.f + i) << std::endl;
		}
	}
	catch (const c10::Error& e)
	{
		std::cerr << e.what() << std::endl;
	}*/

	try
	{
		std::vector<double> avg_diffs(5, 0.f);

		for (size_t i = 0; i < test_iter; i++)
		{
			std::vector<scalar_t> diffs = test_backward();
			std::cout << "index: " << i << std::endl;
			for (size_t j = 0; j < 5; j++)
			{
				avg_diffs[j] += diffs[j];
				std::cout << "avg_diffs[" << j << "]: " << (avg_diffs[j] / (1.f + i)) << std::endl;
			}
			std::cout << std::endl;
		}
	}
	catch (const c10::Error& e)
	{
		std::cerr << e.what() << std::endl;
	}

	return 0;
}