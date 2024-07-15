#!/bin/bash
while true
do
        process_dsupdateserver=`ps -ef|grep zhanhui_fullscreen_good| grep -v grep`
        if [ "$process_dsupdateserver" == "" ]; then
        # 执行python3 /home/avcit/LBK/restart_deepstream.py
                python3 /opt/nvidia/deepstream/deepstream/sources/deepstream_python_apps/apps/zhanhui/zhanhui_fullscreen_good.py &
        fi
        sleep 1
done