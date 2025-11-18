#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <cuda_runtime_api.h>
#include <iostream>
#include <nvbufsurface.h>

#include "gstGetimg.h"
#include "logger.h"

GstGetframe::GstGetframe()
{
}

GstGetframe::~GstGetframe()
{
}

void GstGetframe::init()
{
    pipeline = gst_pipeline_new("rstp-pipeline");
    src = gst_element_factory_make("rtspsrc","src");
    depay = gst_element_factory_make("rtph264depay", "depay");
    parse = gst_element_factory_make("h264parse", "parse");
    decoder = gst_element_factory_make("nvv4l2decoder", "decoder");
    conv = gst_element_factory_make("nvvideoconvert", "conv");
    sink = gst_element_factory_make("appsink", "sink");
}

cv::cuda::GpuMat GstGetframe::wrapNVbufsurface(NvBufSurface *surface)
{
    if (!surface || surface->numFilled == 0)
    {
        customLogger::getInstance()->critical("Invalid NvBufSurface");
    }
    auto &src = surface->surfaceList[0];
    cv::cuda::GpuMat dst(src.height, src.width, CV_8UC3);
    size_t size = dst.step * dst.rows;
    cudaMemcpy(dst.data, src.dataPtr, size, cudaMemcpyDeviceToDevice);
    return dst;
}
gboolean GstGetframe::bus_call(GstBus *bus, GstMessage *msg, gpointer data)
{
    GstElement *pipeline = GST_IS_ELEMENT(data) ? GST_ELEMENT(data) : nullptr;
    if (!pipeline)
    {
        g_printerr("bus_call(): Invalid pipeline pointer.\n");
        return FALSE;
    }

    switch (GST_MESSAGE_TYPE(msg))
    {
    case GST_MESSAGE_ERROR:
    {
        GError *err = nullptr;
        gchar *debug_info = nullptr;
        gst_message_parse_error(msg, &err, &debug_info);

        g_printerr("ERROR from %s: %s\n",
                   GST_OBJECT_NAME(msg->src),
                   err ? err->message : "Unknown error");
        g_printerr("Debug info: %s\n", debug_info ? debug_info : "none");

        // 判斷是否為 fatal error，無法 recover 時直接退出
        if (err &&
            ((err->domain == GST_STREAM_ERROR && err->code == GST_STREAM_ERROR_FAILED) ||
             (err->domain == GST_CORE_ERROR && err->code == GST_CORE_ERROR_MISSING_PLUGIN)))
        {
            g_printerr("Fatal error encountered. Stopping pipeline.\n");
            gst_element_set_state(pipeline, GST_STATE_NULL);
            g_error_free(err);
            g_free(debug_info);
            return FALSE; // 停止主 loop 的 watch
        }

        // 嘗試重啟 pipeline
        g_print("Attempting to restart pipeline...\n");

        GstStateChangeReturn ret;
        ret = gst_element_set_state(pipeline, GST_STATE_NULL);
        if (ret == GST_STATE_CHANGE_FAILURE)
        {
            g_printerr("Failed to set pipeline to NULL.\n");
        }

        ret = gst_element_set_state(pipeline, GST_STATE_READY);
        if (ret == GST_STATE_CHANGE_FAILURE)
        {
            g_printerr("Failed to set pipeline to READY.\n");
        }

        ret = gst_element_set_state(pipeline, GST_STATE_PLAYING);
        if (ret == GST_STATE_CHANGE_FAILURE)
        {
            g_printerr("Failed to set pipeline to PLAYING.\n");
        }

        g_error_free(err);
        g_free(debug_info);
        break;
    }

    case GST_MESSAGE_EOS:
        g_print("End-of-Stream reached.\n");
        gst_element_set_state(pipeline, GST_STATE_NULL);
        return FALSE; // 讓主 loop 結束（可選）

    default:
        break;
    }

    return TRUE;
}


