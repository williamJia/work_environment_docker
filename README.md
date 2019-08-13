# work_environment_docker
工作环境docker

coocaa_work_env --》 酷开测试环境
docker run -v /home/leadtek/jiayuepeng/coocaa_work_env:/root -ti williamjia/coocaa_evn:latest /bin/bash

一.生成相关image镜像

    1. DockerFile 编写
        1. 指定基础环境 FROM python:2.7-alpine
        2. 依赖环境下载 RUN pip install flask Flask-Script
        3. code 相关移植到image内 COPY . /app/
        4. 设置根目录 WORKDIR /app
        5. 暴漏端口设置 EXPOSE 2333
        6. 启动相关app CMD [ "flask", "run", "--host=0.0.0.0", "--port=2333" ]

    2. code 编写
        1. 暴漏端口号指定
        2. ip指定为 0.0.0.0，非127.0.0.1

    3. image 生成
        docker build DockerFile所在目录 (-t tag:指定标签名称)
        docker tag imageId tagname(把上边生成的image打上tag)

    4. image 上传docker hub或者本地压缩打包
        docker hub
        docker tag imageName userName/imageName (打一个标记用于提交)
        docker push williamjia/imageName (williamjia 为docker 的用户名,需提前登陆docker)

二.根据镜像生成容器，运行

    加载image，以下两种方式都可以
        1. 从docker hub上pull下来 （docker image pull williamjia/ai_platform_demo）
        2. 从本地打包好的文件中加载成image

    container 生成 或生成的同时进入docker

        生成：
        docker run -ti -p 80:5000 -v /root:/root -d williamdemo/flask:v1
            1. -p 80:5000 --》指代相应端口，宿主机80端口映射到5000端口
            2.  -d williamdemo/flask:v1 --》指定运行的image名称和版本号
            3. -v /root:/root 宿主机目录和容器的映射，冒号前宿主机目录，冒号后为容器对应目录
            4. 端口访问测试

        生成并进入docker,可以同步上述参数，挂载目录等
            docker run -t -i containerName /bin/bash 该命令可以进入docker容器内部查看