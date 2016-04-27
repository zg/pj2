//******************************************************************************
//
// File:    Cuda.c
// Package: edu.rit.gpu
// Unit:    Class edu.rit.gpu.Cuda
//
// This C source file is copyright (C) 2014 by Alan Kaminsky. All rights
// reserved. For further information, contact the author, Alan Kaminsky, at
// ark@cs.rit.edu.
//
// This C source file is part of the Parallel Java 2 Library ("PJ2"). PJ2 is
// free software; you can redistribute it and/or modify it under the terms of
// the GNU General Public License as published by the Free Software Foundation;
// either version 3 of the License, or (at your option) any later version.
//
// PJ2 is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
// A PARTICULAR PURPOSE. See the GNU General Public License for more details.
//
// A copy of the GNU General Public License is provided in the file gpl.txt. You
// may also obtain a copy of the GNU General Public License on the World Wide
// Web at http://www.gnu.org/licenses/gpl.html.
//
//******************************************************************************

#include <jni.h>
#include <cuda.h>
#include <builtin_types.h>
#include "edu_rit_gpu_Cuda.h"

/**
 * File Cuda.c contains the C code implementations of the native methods in
 * class edu.rit.gpu.Cuda.
 *
 * @author  Alan Kaminsky
 * @version 19-Feb-2014
 */

/*
 * Convert a C pointer to a Java long.
 */
#if defined(__x86_64) || defined(AMD64) || defined(_M_AMD64)
#define cptr_to_jlong(x) ((jlong)x)
#else
#define cptr_to_jlong(x) ((jlong)((jint)x))
#endif

/*
 * Convert a Java long to a C pointer.
 */
#if defined(__x86_64) || defined(AMD64) || defined(_M_AMD64)
#define jlong_to_cptr(x) ((void*)x)
#else
#define jlong_to_cptr(x) ((void*)((jint)x))
#endif

/*
 * Convert a C size_t to a Java long.
 */
#if defined(__x86_64) || defined(AMD64) || defined(_M_AMD64)
#define csize_t_to_jlong(x) ((jlong)x)
#else
#define csize_t_to_jlong(x) ((jlong)((jint)x))
#endif

/*
 * Convert a Java long to a C size_t.
 */
#if defined(__x86_64) || defined(AMD64) || defined(_M_AMD64)
#define jlong_to_csize_t(x) ((size_t)x)
#else
#define jlong_to_csize_t(x) ((size_t)((jint)x))
#endif

/*
 * Convert a C CUdeviceptr to a Java long.
 */
#if defined(__x86_64) || defined(AMD64) || defined(_M_AMD64)
#define CUdeviceptr_to_jlong(x) ((jlong)x)
#else
#define CUdeviceptr_to_jlong(x) ((jlong)((jint)x))
#endif

/*
 * Convert a Java long to a C CUdeviceptr.
 */
#if defined(__x86_64) || defined(AMD64) || defined(_M_AMD64)
#define jlong_to_CUdeviceptr(x) ((CUdeviceptr)x)
#else
#define jlong_to_CUdeviceptr(x) ((CUdeviceptr)((jint)x))
#endif

/*
 * If rv is not CUDA_SUCCESS, throw the appropriate Java exception.
 * Returns 1 if an exception was thrown, 0 otherwise.
 */
int throwCudaException
	(JNIEnv *env,
	 const CUresult rv,
	 const char *msg)
	{
	char *name;
	jclass cls;

	if (rv == CUDA_SUCCESS) return 0;
	switch (rv)
		{
		case CUDA_ERROR_DEINITIALIZED:
			name = "edu/rit/gpu/DeinitializedCudaException";
			break;
		case CUDA_ERROR_FILE_NOT_FOUND:
			name = "edu/rit/gpu/FileNotFoundCudaException";
			break;
		case CUDA_ERROR_INVALID_CONTEXT:
			name = "edu/rit/gpu/InvalidContextCudaException";
			break;
		case CUDA_ERROR_INVALID_DEVICE:
			name = "edu/rit/gpu/InvalidDeviceCudaException";
			break;
		case CUDA_ERROR_INVALID_HANDLE:
			name = "edu/rit/gpu/InvalidHandleCudaException";
			break;
		case CUDA_ERROR_INVALID_IMAGE:
			name = "edu/rit/gpu/InvalidImageCudaException";
			break;
		case CUDA_ERROR_INVALID_VALUE:
			name = "edu/rit/gpu/InvalidValueCudaException";
			break;
		case CUDA_ERROR_LAUNCH_FAILED:
			name = "edu/rit/gpu/LaunchFailedCudaException";
			break;
		case CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING:
			name = "edu/rit/gpu/LaunchIncompatibleTexturingCudaException";
			break;
		case CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES:
			name = "edu/rit/gpu/LaunchOutOfResourcesCudaException";
			break;
		case CUDA_ERROR_LAUNCH_TIMEOUT:
			name = "edu/rit/gpu/LaunchTimeoutCudaException";
			break;
		case CUDA_ERROR_NOT_FOUND:
			name = "edu/rit/gpu/NotFoundCudaException";
			break;
		case CUDA_ERROR_NOT_INITIALIZED:
			name = "edu/rit/gpu/NotInitializedCudaException";
			break;
		case CUDA_ERROR_OUT_OF_MEMORY:
			name = "edu/rit/gpu/OutOfMemoryCudaException";
			break;
		case CUDA_ERROR_SHARED_OBJECT_INIT_FAILED:
			name = "edu/rit/gpu/SharedObjectInitFailedCudaException";
			break;
		case CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND:
			name = "edu/rit/gpu/SharedObjectSymbolNotFoundCudaException";
			break;
		case CUDA_ERROR_UNKNOWN:
			name = "edu/rit/gpu/UnknownCudaException";
			break;
		default:
			name = "edu/rit/gpu/CudaException";
			break;
		}
	cls = (*env)->FindClass (env, name);
	if (cls != NULL)
		(*env)->ThrowNew (env, cls, msg);
	(*env)->DeleteLocalRef (env, cls);
	return 1;
	}

/*
 * Class:     edu_rit_gpu_Cuda
 * Method:    is64BitPointer
 * Signature: ()Z
 */
JNIEXPORT jboolean JNICALL Java_edu_rit_gpu_Cuda_is64BitPointer
	(JNIEnv *env,
	 jclass cls)
	{
#if defined(__x86_64) || defined(AMD64) || defined(_M_AMD64)
	return 1;
#else
	return 0;
#endif
	}

/*
 * Class:     edu_rit_gpu_Cuda
 * Method:    cuInit
 * Signature: (I)V
 */
JNIEXPORT void JNICALL Java_edu_rit_gpu_Cuda_cuInit
	(JNIEnv *env,
	 jclass cls,
	 jint flags)
	{
	CUresult rv = cuInit (flags);
	throwCudaException (env, rv, "calling cuInit()");
	}

/*
 * Class:     edu_rit_gpu_Cuda
 * Method:    cuDeviceGetCount
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_edu_rit_gpu_Cuda_cuDeviceGetCount
	(JNIEnv *env,
	 jclass cls)
	{
	jint count = 0;
	CUresult rv = cuDeviceGetCount (&count);
	throwCudaException (env, rv, "calling cuDeviceGetCount()");
	return count;
	}

/*
 * Class:     edu_rit_gpu_Cuda
 * Method:    cuDeviceGet
 * Signature: (I)I
 */
JNIEXPORT jint JNICALL Java_edu_rit_gpu_Cuda_cuDeviceGet
	(JNIEnv *env,
	 jclass cls,
	 jint ordinal)
	{
	CUdevice device = 0;
	CUresult rv = cuDeviceGet (&device, ordinal);
	throwCudaException (env, rv, "calling cuDeviceGet()");
	return device;
	}

/*
 * Class:     edu_rit_gpu_Cuda
 * Method:    cuDeviceGetName
 * Signature: (I)Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL Java_edu_rit_gpu_Cuda_cuDeviceGetName
	(JNIEnv *env,
	 jclass cls,
	 jint device)
	{
	char name [128];
	name[0] = '\0';
	CUresult rv = cuDeviceGetName (name, 127, device);
	throwCudaException (env, rv, "calling cuDeviceGetName()");
	return (*env)->NewStringUTF (env, name);
	}

/*
 * Class:     edu_rit_gpu_Cuda
 * Method:    cuDeviceGetAttributeComputeCapabilityMajor
 * Signature: (I)Ljava/lang/String;
 */
JNIEXPORT jint JNICALL Java_edu_rit_gpu_Cuda_cuDeviceGetAttributeComputeCapabilityMajor
	(JNIEnv *env,
	 jclass cls,
	 jint device)
	{
	jint pi = 0;
	CUresult rv = cuDeviceGetAttribute
		(&pi, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
	throwCudaException (env, rv, "calling cuDeviceGetAttribute()");
	return pi;
	}

/*
 * Class:     edu_rit_gpu_Cuda
 * Method:    cuDeviceGetAttributeComputeCapabilityMinor
 * Signature: (I)Ljava/lang/String;
 */
JNIEXPORT jint JNICALL Java_edu_rit_gpu_Cuda_cuDeviceGetAttributeComputeCapabilityMinor
	(JNIEnv *env,
	 jclass cls,
	 jint device)
	{
	jint pi = 0;
	CUresult rv = cuDeviceGetAttribute
		(&pi, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device);
	throwCudaException (env, rv, "calling cuDeviceGetAttribute()");
	return pi;
	}

/*
 * Class:     edu_rit_gpu_Cuda
 * Method:    cuDeviceGetAttributeMultiprocessorCount
 * Signature: (I)I
 */
JNIEXPORT jint JNICALL Java_edu_rit_gpu_Cuda_cuDeviceGetAttributeMultiprocessorCount
	(JNIEnv *env,
	 jclass cls,
	 jint device)
	{
	jint pi = 0;
	CUresult rv = cuDeviceGetAttribute
		(&pi, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device);
	throwCudaException (env, rv, "calling cuDeviceGetAttribute()");
	return pi;
	}

/*
 * Class:     edu_rit_gpu_Cuda
 * Method:    cuCtxCreate
 * Signature: (II)J
 */
JNIEXPORT jlong JNICALL Java_edu_rit_gpu_Cuda_cuCtxCreate
	(JNIEnv *env,
	 jclass cls,
	 jint flags,
	 jint device)
	{
	CUcontext ctx = NULL;
	CUresult rv = cuCtxCreate (&ctx, flags, device);
	throwCudaException (env, rv, "calling cuCtxCreate()");
	return cptr_to_jlong (ctx);
	}

/*
 * Class:     edu_rit_gpu_Cuda
 * Method:    cuCtxDestroy
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_edu_rit_gpu_Cuda_cuCtxDestroy
	(JNIEnv *env,
	 jclass cls,
	 jlong ctx)
	{
	CUresult rv = cuCtxDestroy ((CUcontext) jlong_to_cptr (ctx));
	throwCudaException (env, rv, "calling cuCtxDestroy()");
	}

/*
 * Class:     edu_rit_gpu_Cuda
 * Method:    cuCtxSetCacheConfig
 * Signature: (JI)V
 */
JNIEXPORT void JNICALL Java_edu_rit_gpu_Cuda_cuCtxSetCacheConfig
	(JNIEnv *env,
	 jclass cls,
	 jlong ctx,
	 jint config)
	{
	CUcontext context = (CUcontext) jlong_to_cptr (ctx);
	CUresult rv;
	int exc = 0;
	rv = cuCtxPushCurrent (context);
	exc = throwCudaException (env, rv, "calling cuCtxPushCurrent()");
	if (! exc)
		{
		rv = cuCtxSetCacheConfig (config);
		exc = throwCudaException (env, rv, "calling cuCtxSetCacheConfig()");
		rv = cuCtxPopCurrent (&context);
		if (! exc)
			throwCudaException (env, rv, "calling cuCtxPopCurrent()");
		}
	}

/*
 * Class:     edu_rit_gpu_Cuda
 * Method:    cuCtxGetCacheConfig
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_edu_rit_gpu_Cuda_cuCtxGetCacheConfig
	(JNIEnv *env,
	 jclass cls,
	 jlong ctx)
	{
	CUcontext context = (CUcontext) jlong_to_cptr (ctx);
	CUfunc_cache config;
	CUresult rv;
	int exc = 0;
	rv = cuCtxPushCurrent (context);
	exc = throwCudaException (env, rv, "calling cuCtxPushCurrent()");
	if (! exc)
		{
		rv = cuCtxGetCacheConfig (&config);
		exc = throwCudaException (env, rv, "calling cuCtxGetCacheConfig()");
		rv = cuCtxPopCurrent (&context);
		if (! exc)
			throwCudaException (env, rv, "calling cuCtxPopCurrent()");
		}
	return config;
	}

/*
 * Class:     edu_rit_gpu_Cuda
 * Method:    cuMemAlloc
 * Signature: (JJ)J
 */
JNIEXPORT jlong JNICALL Java_edu_rit_gpu_Cuda_cuMemAlloc
	(JNIEnv *env,
	 jclass cls,
	 jlong ctx,
	 jlong bytesize)
	{
	CUcontext context = (CUcontext) jlong_to_cptr (ctx);
	size_t N = jlong_to_csize_t (bytesize);
	CUdeviceptr dptr = 0;
	CUresult rv;
	int exc = 0;
	rv = cuCtxPushCurrent (context);
	exc = throwCudaException (env, rv, "calling cuCtxPushCurrent()");
	if (! exc)
		{
		rv = cuMemAlloc (&dptr, N);
		exc = throwCudaException (env, rv, "calling cuMemAlloc()");
		rv = cuCtxPopCurrent (&context);
		if (! exc)
			throwCudaException (env, rv, "calling cuCtxPopCurrent()");
		}
	return CUdeviceptr_to_jlong (dptr);
	}

/*
 * Class:     edu_rit_gpu_Cuda
 * Method:    cuMemFree
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_edu_rit_gpu_Cuda_cuMemFree
	(JNIEnv *env,
	 jclass cls,
	 jlong ctx,
	 jlong dptr)
	{
	CUcontext context = (CUcontext) jlong_to_cptr (ctx);
	CUdeviceptr ptr = jlong_to_CUdeviceptr (dptr);
	CUresult rv;
	int exc = 0;
	rv = cuCtxPushCurrent (context);
	exc = throwCudaException (env, rv, "calling cuCtxPushCurrent()");
	if (! exc)
		{
		rv = cuMemFree (ptr);
		exc = throwCudaException (env, rv, "calling cuMemFree()");
		rv = cuCtxPopCurrent (&context);
		if (! exc)
			throwCudaException (env, rv, "calling cuCtxPopCurrent()");
		}
	}

/*
 * Class:     edu_rit_gpu_Cuda
 * Method:    cuMemcpyHtoD
 * Signature: (JJI[BII)V
 */
JNIEXPORT void JNICALL Java_edu_rit_gpu_Cuda_cuMemcpyHtoD__JJI_3BII
	(JNIEnv *env,
	 jclass cls,
	 jlong ctx,
	 jlong dst,
	 jint dstindex,
	 jbyteArray src,
	 jint srcindex,
	 jint nelem)
	{
	CUcontext context = (CUcontext) jlong_to_cptr (ctx);
	CUdeviceptr dptr = jlong_to_CUdeviceptr (dst + dstindex*sizeof(jbyte));
	size_t N = nelem*sizeof(jbyte);
	CUresult rv;
	int exc = 0;
	jbyte *csrc = (*env)->GetByteArrayElements (env, src, NULL);
	if (csrc == NULL) return;
	rv = cuCtxPushCurrent (context);
	exc = throwCudaException (env, rv, "calling cuCtxPushCurrent()");
	if (! exc)
		{
		rv = cuMemcpyHtoD (dptr, csrc + srcindex, N);
		exc = throwCudaException (env, rv, "calling cuMemcpyHtoD()");
		rv = cuCtxPopCurrent (&context);
		if (! exc)
			throwCudaException (env, rv, "calling cuCtxPopCurrent()");
		}
	(*env)->ReleaseByteArrayElements (env, src, csrc, 0);
	}

/*
 * Class:     edu_rit_gpu_Cuda
 * Method:    cuMemcpyDtoH
 * Signature: (J[BIJII)V
 */
JNIEXPORT void JNICALL Java_edu_rit_gpu_Cuda_cuMemcpyDtoH__J_3BIJII
	(JNIEnv *env,
	 jclass cls,
	 jlong ctx,
	 jbyteArray dst,
	 jint dstindex,
	 jlong src,
	 jint srcindex,
	 jint nelem)
	{
	CUcontext context = (CUcontext) jlong_to_cptr (ctx);
	CUdeviceptr dptr = jlong_to_CUdeviceptr (src + srcindex*sizeof(jbyte));
	size_t N = nelem*sizeof(jbyte);
	CUresult rv;
	int exc = 0;
	jbyte *cdst = (*env)->GetByteArrayElements (env, dst, NULL);
	if (cdst == NULL) return;
	rv = cuCtxPushCurrent (context);
	exc = throwCudaException (env, rv, "calling cuCtxPushCurrent()");
	if (! exc)
		{
		rv = cuMemcpyDtoH (cdst + dstindex, dptr, N);
		exc = throwCudaException (env, rv, "calling cuMemcpyDtoH()");
		rv = cuCtxPopCurrent (&context);
		if (! exc)
			throwCudaException (env, rv, "calling cuCtxPopCurrent()");
		}
	(*env)->ReleaseByteArrayElements (env, dst, cdst, 0);
	}

/*
 * Class:     edu_rit_gpu_Cuda
 * Method:    cuMemcpyHtoD
 * Signature: (JJI[SII)V
 */
JNIEXPORT void JNICALL Java_edu_rit_gpu_Cuda_cuMemcpyHtoD__JJI_3SII
	(JNIEnv *env,
	 jclass cls,
	 jlong ctx,
	 jlong dst,
	 jint dstindex,
	 jshortArray src,
	 jint srcindex,
	 jint nelem)
	{
	CUcontext context = (CUcontext) jlong_to_cptr (ctx);
	CUdeviceptr dptr = jlong_to_CUdeviceptr (dst + dstindex*sizeof(jshort));
	size_t N = nelem*sizeof(jshort);
	CUresult rv;
	int exc = 0;
	jshort *csrc = (*env)->GetShortArrayElements (env, src, NULL);
	if (csrc == NULL) return;
	rv = cuCtxPushCurrent (context);
	exc = throwCudaException (env, rv, "calling cuCtxPushCurrent()");
	if (! exc)
		{
		rv = cuMemcpyHtoD (dptr, csrc + srcindex, N);
		exc = throwCudaException (env, rv, "calling cuMemcpyHtoD()");
		rv = cuCtxPopCurrent (&context);
		if (! exc)
			throwCudaException (env, rv, "calling cuCtxPopCurrent()");
		}
	(*env)->ReleaseShortArrayElements (env, src, csrc, 0);
	}

/*
 * Class:     edu_rit_gpu_Cuda
 * Method:    cuMemcpyDtoH
 * Signature: (J[SIJII)V
 */
JNIEXPORT void JNICALL Java_edu_rit_gpu_Cuda_cuMemcpyDtoH__J_3SIJII
	(JNIEnv *env,
	 jclass cls,
	 jlong ctx,
	 jshortArray dst,
	 jint dstindex,
	 jlong src,
	 jint srcindex,
	 jint nelem)
	{
	CUcontext context = (CUcontext) jlong_to_cptr (ctx);
	CUdeviceptr dptr = jlong_to_CUdeviceptr (src + srcindex*sizeof(jshort));
	size_t N = nelem*sizeof(jshort);
	CUresult rv;
	int exc = 0;
	jshort *cdst = (*env)->GetShortArrayElements (env, dst, NULL);
	if (cdst == NULL) return;
	rv = cuCtxPushCurrent (context);
	exc = throwCudaException (env, rv, "calling cuCtxPushCurrent()");
	if (! exc)
		{
		rv = cuMemcpyDtoH (cdst + dstindex, dptr, N);
		exc = throwCudaException (env, rv, "calling cuMemcpyDtoH()");
		rv = cuCtxPopCurrent (&context);
		if (! exc)
			throwCudaException (env, rv, "calling cuCtxPopCurrent()");
		}
	(*env)->ReleaseShortArrayElements (env, dst, cdst, 0);
	}

/*
 * Class:     edu_rit_gpu_Cuda
 * Method:    cuMemcpyHtoD
 * Signature: (JJI[III)V
 */
JNIEXPORT void JNICALL Java_edu_rit_gpu_Cuda_cuMemcpyHtoD__JJI_3III
	(JNIEnv *env,
	 jclass cls,
	 jlong ctx,
	 jlong dst,
	 jint dstindex,
	 jintArray src,
	 jint srcindex,
	 jint nelem)
	{
	CUcontext context = (CUcontext) jlong_to_cptr (ctx);
	CUdeviceptr dptr = jlong_to_CUdeviceptr (dst + dstindex*sizeof(jint));
	size_t N = nelem*sizeof(jint);
	CUresult rv;
	int exc = 0;
	jint *csrc = (*env)->GetIntArrayElements (env, src, NULL);
	if (csrc == NULL) return;
	rv = cuCtxPushCurrent (context);
	exc = throwCudaException (env, rv, "calling cuCtxPushCurrent()");
	if (! exc)
		{
		rv = cuMemcpyHtoD (dptr, csrc + srcindex, N);
		exc = throwCudaException (env, rv, "calling cuMemcpyHtoD()");
		rv = cuCtxPopCurrent (&context);
		if (! exc)
			throwCudaException (env, rv, "calling cuCtxPopCurrent()");
		}
	(*env)->ReleaseIntArrayElements (env, src, csrc, 0);
	}

/*
 * Class:     edu_rit_gpu_Cuda
 * Method:    cuMemcpyDtoH
 * Signature: (J[IIJII)V
 */
JNIEXPORT void JNICALL Java_edu_rit_gpu_Cuda_cuMemcpyDtoH__J_3IIJII
	(JNIEnv *env,
	 jclass cls,
	 jlong ctx,
	 jintArray dst,
	 jint dstindex,
	 jlong src,
	 jint srcindex,
	 jint nelem)
	{
	CUcontext context = (CUcontext) jlong_to_cptr (ctx);
	CUdeviceptr dptr = jlong_to_CUdeviceptr (src + srcindex*sizeof(jint));
	size_t N = nelem*sizeof(jint);
	CUresult rv;
	int exc = 0;
	jint *cdst = (*env)->GetIntArrayElements (env, dst, NULL);
	if (cdst == NULL) return;
	rv = cuCtxPushCurrent (context);
	exc = throwCudaException (env, rv, "calling cuCtxPushCurrent()");
	if (! exc)
		{
		rv = cuMemcpyDtoH (cdst + dstindex, dptr, N);
		exc = throwCudaException (env, rv, "calling cuMemcpyDtoH()");
		rv = cuCtxPopCurrent (&context);
		if (! exc)
			throwCudaException (env, rv, "calling cuCtxPopCurrent()");
		}
	(*env)->ReleaseIntArrayElements (env, dst, cdst, 0);
	}

/*
 * Class:     edu_rit_gpu_Cuda
 * Method:    cuMemcpyHtoD
 * Signature: (JJI[JII)V
 */
JNIEXPORT void JNICALL Java_edu_rit_gpu_Cuda_cuMemcpyHtoD__JJI_3JII
	(JNIEnv *env,
	 jclass cls,
	 jlong ctx,
	 jlong dst,
	 jint dstindex,
	 jlongArray src,
	 jint srcindex,
	 jint nelem)
	{
	CUcontext context = (CUcontext) jlong_to_cptr (ctx);
	CUdeviceptr dptr = jlong_to_CUdeviceptr (dst + dstindex*sizeof(jlong));
	size_t N = nelem*sizeof(jlong);
	CUresult rv;
	int exc = 0;
	jlong *csrc = (*env)->GetLongArrayElements (env, src, NULL);
	if (csrc == NULL) return;
	rv = cuCtxPushCurrent (context);
	exc = throwCudaException (env, rv, "calling cuCtxPushCurrent()");
	if (! exc)
		{
		rv = cuMemcpyHtoD (dptr, csrc + srcindex, N);
		exc = throwCudaException (env, rv, "calling cuMemcpyHtoD()");
		rv = cuCtxPopCurrent (&context);
		if (! exc)
			throwCudaException (env, rv, "calling cuCtxPopCurrent()");
		}
	(*env)->ReleaseLongArrayElements (env, src, csrc, 0);
	}

/*
 * Class:     edu_rit_gpu_Cuda
 * Method:    cuMemcpyDtoH
 * Signature: (J[JIJII)V
 */
JNIEXPORT void JNICALL Java_edu_rit_gpu_Cuda_cuMemcpyDtoH__J_3JIJII
	(JNIEnv *env,
	 jclass cls,
	 jlong ctx,
	 jlongArray dst,
	 jint dstindex,
	 jlong src,
	 jint srcindex,
	 jint nelem)
	{
	CUcontext context = (CUcontext) jlong_to_cptr (ctx);
	CUdeviceptr dptr = jlong_to_CUdeviceptr (src + srcindex*sizeof(jlong));
	size_t N = nelem*sizeof(jlong);
	CUresult rv;
	int exc = 0;
	jlong *cdst = (*env)->GetLongArrayElements (env, dst, NULL);
	if (cdst == NULL) return;
	rv = cuCtxPushCurrent (context);
	exc = throwCudaException (env, rv, "calling cuCtxPushCurrent()");
	if (! exc)
		{
		rv = cuMemcpyDtoH (cdst + dstindex, dptr, N);
		exc = throwCudaException (env, rv, "calling cuMemcpyDtoH()");
		rv = cuCtxPopCurrent (&context);
		if (! exc)
			throwCudaException (env, rv, "calling cuCtxPopCurrent()");
		}
	(*env)->ReleaseLongArrayElements (env, dst, cdst, 0);
	}

/*
 * Class:     edu_rit_gpu_Cuda
 * Method:    cuMemcpyHtoD
 * Signature: (JJI[FII)V
 */
JNIEXPORT void JNICALL Java_edu_rit_gpu_Cuda_cuMemcpyHtoD__JJI_3FII
	(JNIEnv *env,
	 jclass cls,
	 jlong ctx,
	 jlong dst,
	 jint dstindex,
	 jfloatArray src,
	 jint srcindex,
	 jint nelem)
	{
	CUcontext context = (CUcontext) jlong_to_cptr (ctx);
	CUdeviceptr dptr = jlong_to_CUdeviceptr (dst + dstindex*sizeof(jfloat));
	size_t N = nelem*sizeof(jfloat);
	CUresult rv;
	int exc = 0;
	jfloat *csrc = (*env)->GetFloatArrayElements (env, src, NULL);
	if (csrc == NULL) return;
	rv = cuCtxPushCurrent (context);
	exc = throwCudaException (env, rv, "calling cuCtxPushCurrent()");
	if (! exc)
		{
		rv = cuMemcpyHtoD (dptr, csrc + srcindex, N);
		exc = throwCudaException (env, rv, "calling cuMemcpyHtoD()");
		rv = cuCtxPopCurrent (&context);
		if (! exc)
			throwCudaException (env, rv, "calling cuCtxPopCurrent()");
		}
	(*env)->ReleaseFloatArrayElements (env, src, csrc, 0);
	}

/*
 * Class:     edu_rit_gpu_Cuda
 * Method:    cuMemcpyDtoH
 * Signature: (J[FIJII)V
 */
JNIEXPORT void JNICALL Java_edu_rit_gpu_Cuda_cuMemcpyDtoH__J_3FIJII
	(JNIEnv *env,
	 jclass cls,
	 jlong ctx,
	 jfloatArray dst,
	 jint dstindex,
	 jlong src,
	 jint srcindex,
	 jint nelem)
	{
	CUcontext context = (CUcontext) jlong_to_cptr (ctx);
	CUdeviceptr dptr = jlong_to_CUdeviceptr (src + srcindex*sizeof(jfloat));
	size_t N = nelem*sizeof(jfloat);
	CUresult rv;
	int exc = 0;
	jfloat *cdst = (*env)->GetFloatArrayElements (env, dst, NULL);
	if (cdst == NULL) return;
	rv = cuCtxPushCurrent (context);
	exc = throwCudaException (env, rv, "calling cuCtxPushCurrent()");
	if (! exc)
		{
		rv = cuMemcpyDtoH (cdst + dstindex, dptr, N);
		exc = throwCudaException (env, rv, "calling cuMemcpyDtoH()");
		rv = cuCtxPopCurrent (&context);
		if (! exc)
			throwCudaException (env, rv, "calling cuCtxPopCurrent()");
		}
	(*env)->ReleaseFloatArrayElements (env, dst, cdst, 0);
	}

/*
 * Class:     edu_rit_gpu_Cuda
 * Method:    cuMemcpyHtoD
 * Signature: (JJI[DII)V
 */
JNIEXPORT void JNICALL Java_edu_rit_gpu_Cuda_cuMemcpyHtoD__JJI_3DII
	(JNIEnv *env,
	 jclass cls,
	 jlong ctx,
	 jlong dst,
	 jint dstindex,
	 jdoubleArray src,
	 jint srcindex,
	 jint nelem)
	{
	CUcontext context = (CUcontext) jlong_to_cptr (ctx);
	CUdeviceptr dptr = jlong_to_CUdeviceptr (dst + dstindex*sizeof(jdouble));
	size_t N = nelem*sizeof(jdouble);
	CUresult rv;
	int exc = 0;
	jdouble *csrc = (*env)->GetDoubleArrayElements (env, src, NULL);
	if (csrc == NULL) return;
	rv = cuCtxPushCurrent (context);
	exc = throwCudaException (env, rv, "calling cuCtxPushCurrent()");
	if (! exc)
		{
		rv = cuMemcpyHtoD (dptr, csrc + srcindex, N);
		exc = throwCudaException (env, rv, "calling cuMemcpyHtoD()");
		rv = cuCtxPopCurrent (&context);
		if (! exc)
			throwCudaException (env, rv, "calling cuCtxPopCurrent()");
		}
	(*env)->ReleaseDoubleArrayElements (env, src, csrc, 0);
	}

/*
 * Class:     edu_rit_gpu_Cuda
 * Method:    cuMemcpyDtoH
 * Signature: (J[DIJII)V
 */
JNIEXPORT void JNICALL Java_edu_rit_gpu_Cuda_cuMemcpyDtoH__J_3DIJII
	(JNIEnv *env,
	 jclass cls,
	 jlong ctx,
	 jdoubleArray dst,
	 jint dstindex,
	 jlong src,
	 jint srcindex,
	 jint nelem)
	{
	CUcontext context = (CUcontext) jlong_to_cptr (ctx);
	CUdeviceptr dptr = jlong_to_CUdeviceptr (src + srcindex*sizeof(jdouble));
	size_t N = nelem*sizeof(jdouble);
	CUresult rv;
	int exc = 0;
	jdouble *cdst = (*env)->GetDoubleArrayElements (env, dst, NULL);
	if (cdst == NULL) return;
	rv = cuCtxPushCurrent (context);
	exc = throwCudaException (env, rv, "calling cuCtxPushCurrent()");
	if (! exc)
		{
		rv = cuMemcpyDtoH (cdst + dstindex, dptr, N);
		exc = throwCudaException (env, rv, "calling cuMemcpyDtoH()");
		rv = cuCtxPopCurrent (&context);
		if (! exc)
			throwCudaException (env, rv, "calling cuCtxPopCurrent()");
		}
	(*env)->ReleaseDoubleArrayElements (env, dst, cdst, 0);
	}

/*
 * Class:     edu_rit_gpu_Cuda
 * Method:    cuModuleLoad
 * Signature: (JLjava/lang/String;)J
 */
JNIEXPORT jlong JNICALL Java_edu_rit_gpu_Cuda_cuModuleLoad
	(JNIEnv *env,
	 jclass cls,
	 jlong ctx,
	 jstring fname)
	{
	CUcontext context = (CUcontext) jlong_to_cptr (ctx);
	CUresult rv;
	int exc = 0;
	CUmodule module = NULL;
	const jbyte *cfname = (*env)->GetStringUTFChars (env, fname, NULL);
	if (cfname == NULL) return 0;
	rv = cuCtxPushCurrent (context);
	exc = throwCudaException (env, rv, "calling cuCtxPushCurrent()");
	if (! exc)
		{
		rv = cuModuleLoad (&module, cfname);
		exc = throwCudaException (env, rv, "calling cuModuleLoad()");
		rv = cuCtxPopCurrent (&context);
		if (! exc)
			throwCudaException (env, rv, "calling cuCtxPopCurrent()");
		}
	(*env)->ReleaseStringUTFChars (env, fname, cfname);
	return cptr_to_jlong (module);
	}

/*
 * Class:     edu_rit_gpu_Cuda
 * Method:    cuModuleUnload
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_edu_rit_gpu_Cuda_cuModuleUnload
	(JNIEnv * env,
	 jclass cls,
	 jlong ctx,
	 jlong mod)
	{
	CUcontext context = (CUcontext) jlong_to_cptr (ctx);
	CUmodule module = (CUmodule) jlong_to_cptr (mod);
	CUresult rv;
	int exc = 0;
	rv = cuCtxPushCurrent (context);
	exc = throwCudaException (env, rv, "calling cuCtxPushCurrent()");
	if (! exc)
		{
		rv = cuModuleUnload (module);
		exc = throwCudaException (env, rv, "calling cuModuleUnload()");
		rv = cuCtxPopCurrent (&context);
		if (! exc)
			throwCudaException (env, rv, "calling cuCtxPopCurrent()");
		}
	}

/*
 * Class:     edu_rit_gpu_Cuda
 * Method:    cuModuleGetFunction
 * Signature: (JJLjava/lang/String;)J
 */
JNIEXPORT jlong JNICALL Java_edu_rit_gpu_Cuda_cuModuleGetFunction
	(JNIEnv *env,
	 jclass cls,
	 jlong ctx,
	 jlong mod,
	 jstring name)
	{
	CUcontext context = (CUcontext) jlong_to_cptr (ctx);
	CUmodule module = (CUmodule) jlong_to_cptr (mod);
	CUresult rv;
	int exc = 0;
	CUfunction function = NULL;
	const jbyte *cname = (*env)->GetStringUTFChars (env, name, NULL);
	if (cname == NULL) return 0;
	rv = cuCtxPushCurrent (context);
	exc = throwCudaException (env, rv, "calling cuCtxPushCurrent()");
	if (! exc)
		{
		rv = cuModuleGetFunction (&function, module, cname);
		exc = throwCudaException (env, rv, "calling cuModuleGetFunction()");
		rv = cuCtxPopCurrent (&context);
		if (! exc)
			throwCudaException (env, rv, "calling cuCtxPopCurrent()");
		}
	(*env)->ReleaseStringUTFChars (env, name, cname);
	return cptr_to_jlong (function);
	}

/*
 * Class:     edu_rit_gpu_Cuda
 * Method:    cuModuleGetGlobal
 * Signature: (JJLjava/lang/String;)J
 */
JNIEXPORT jlong JNICALL Java_edu_rit_gpu_Cuda_cuModuleGetGlobal
	(JNIEnv *env,
	 jclass cls,
	 jlong ctx,
	 jlong mod,
	 jstring name)
	{
	CUcontext context = (CUcontext) jlong_to_cptr (ctx);
	CUmodule module = (CUmodule) jlong_to_cptr (mod);
	CUresult rv;
	int exc = 0;
	CUdeviceptr dptr = 0;
	const jbyte *cname = (*env)->GetStringUTFChars (env, name, NULL);
	if (cname == NULL) return 0;
	rv = cuCtxPushCurrent (context);
	exc = throwCudaException (env, rv, "calling cuCtxPushCurrent()");
	if (! exc)
		{
		rv = cuModuleGetGlobal (&dptr, NULL, module, cname);
		exc = throwCudaException (env, rv, "calling cuModuleGetGlobal()");
		rv = cuCtxPopCurrent (&context);
		if (! exc)
			throwCudaException (env, rv, "calling cuCtxPopCurrent()");
		}
	(*env)->ReleaseStringUTFChars (env, name, cname);
	return CUdeviceptr_to_jlong (dptr);
	}

/*
 * Class:     edu_rit_gpu_Cuda
 * Method:    cuFuncSetCacheConfig
 * Signature: (JJI)V
 */
JNIEXPORT void JNICALL Java_edu_rit_gpu_Cuda_cuFuncSetCacheConfig
	(JNIEnv *env,
	 jclass cls,
	 jlong ctx,
	 jlong func,
	 jint config)
	{
	CUcontext context = (CUcontext) jlong_to_cptr (ctx);
	CUfunction function = (CUfunction) jlong_to_cptr (func);
	CUresult rv;
	int exc = 0;
	rv = cuCtxPushCurrent (context);
	exc = throwCudaException (env, rv, "calling cuCtxPushCurrent()");
	if (! exc)
		{
		rv = cuFuncSetCacheConfig (function, config);
		exc = throwCudaException (env, rv, "calling cuFuncSetCacheConfig()");
		rv = cuCtxPopCurrent (&context);
		if (! exc)
			throwCudaException (env, rv, "calling cuCtxPopCurrent()");
		}
	}

/*
 * Class:     edu_rit_gpu_Cuda
 * Method:    cuLaunchKernel
 * Signature: (JJIIIIIIII[B[B)V
 */
JNIEXPORT void JNICALL Java_edu_rit_gpu_Cuda_cuLaunchKernel
	(JNIEnv *env,
	 jclass cls,
	 jlong ctx,
	 jlong func,
	 jint gridDimX,
	 jint gridDimY,
	 jint gridDimZ,
	 jint blockDimX,
	 jint blockDimY,
	 jint blockDimZ,
	 jint sharedMemBytes,
	 jint argc,
	 jbyteArray argv,
	 jbyteArray argp)
	{
	CUcontext context = (CUcontext) jlong_to_cptr (ctx);
	CUfunction function = (CUfunction) jlong_to_cptr (func);
	CUresult rv;
	int exc = 0;
	int i;

	// Set up pointers to argument values.
	jbyte *cargv = (*env)->GetByteArrayElements (env, argv, NULL);
	jbyte *cargp = (*env)->GetByteArrayElements (env, argp, NULL);
	jbyte *src = cargv;
	void **dst = (void**)cargp;
	for (i = 0; i < argc; ++ i, src += 8, ++ dst)
		*dst = (void*)src;

	// Invoke kernel.
	rv = cuCtxPushCurrent (context);
	exc = throwCudaException (env, rv, "calling cuCtxPushCurrent()");
	if (! exc)
		{
		rv = cuLaunchKernel (function, gridDimX, gridDimY, gridDimZ,
			blockDimX, blockDimY, blockDimZ, sharedMemBytes,
			/*hStream*/ NULL, /*kernelParams*/ (void**)cargp, 
			/*extra*/ NULL);
		exc = throwCudaException (env, rv, "calling cuLaunchKernel()");
		if (! exc)
			{
			rv = cuCtxSynchronize();
			exc = throwCudaException (env, rv, "calling cuCtxSynchronize()");
			}
		rv = cuCtxPopCurrent (&context);
		if (! exc)
			throwCudaException (env, rv, "calling cuCtxPopCurrent()");
		}

	// Release storage.
	(*env)->ReleaseByteArrayElements (env, argv, cargv, JNI_ABORT);
	(*env)->ReleaseByteArrayElements (env, argp, cargp, JNI_ABORT);
	}
