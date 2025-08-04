#ifndef GSTGETIMG_H
#define GSTGETIMG_H
#pragma once

#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <cuda_runtime_api.h>
#include <iostream>
#include <nvbufsurface.h>

class GstGetframe
{
private:
    cv::cuda::GpuMat wrapNVbufsurface(NvBufSurface *surface);
    static gboolean bus_call(GstBus *bus, GstMessage *msg, gpointer data);
    GstElement *pipeline;
    GstElement *src;
    GstElement *depay;
    GstElement *parse;
    GstElement *decoder;
    GstElement *conv;
    GstElement *sink;

public:
    GstGetframe();
    ~GstGetframe();
    void init();
};

#endif