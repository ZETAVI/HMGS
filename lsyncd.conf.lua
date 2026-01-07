-- lsyncd (Live Syncing Daemon) 是 Linux 系统运维中处理此类“实时镜像”需求的瑞士军刀

-- 它的工作原理是：监听 (Watch) + 聚合 (Aggregate) + 执行 (Execute)。
--    1. 监听：它利用 Linux 内核的 inotify 机制，监控 xxx 文件夹下的每一个文件变动（写入、创建、移动）。
--    2. 聚合：它不会因为你写了一行日志就马上触发传输（那样效率太低），而是会根据配置的 delay
--       时间，把几秒内的变动“攒”在一起。
--    3. 执行：调用 rsync 命令，一次性把这几秒的变动增量同步过去。

-- `lsyncd` 的默认行为设计就是为了应对“中途上车”的情况
--  Lsyncd 的启动流程：先全量，后增量
--   当你敲下启动命令的那一刻，lsyncd 实际上分为两个阶段工作：
--    * 阶段一：启动时全量扫描 (Startup Phase)
--        * lsyncd 启动后的第一件事，不是等待变化，而是立刻触发一次完整的 `rsync`。
--        * 它会扫描 Server A 的 xxx 目录，对比 Server B 的目录。
--        * 结果：你之前训练生成的那些“旧”文件，以及当前正在写的文件的“当前状态”，都会在这个阶段被传输到
--          Server B。它会自动“追平”进度。
--    * 阶段二：进入监听模式 (Normal Operation)
--        * 全量同步完成后，它才会正式进入 inotify 监听模式。
--        * 此后，只有当文件发生变化（新增、修改）时，才会触发后续的同步。



-- 具体使用：
-- # 作为当前用户运行（如果 SSH key 是配在当前用户下的）
-- lsyncd ~/lsyncd.conf.lua
-- # 或者作为系统服务运行（需要把配置文件放到 /etc/lsyncd.conf 并确保 root 有 SSH key）
-- sudo service lsyncd start


settings {
    -- 日志文件位置，出了问题看这里
    logfile = "/tmp/lsyncd.log",
    -- 状态文件，记录当前监控了多少目录，状态如何
    statusFile = "/tmp/lsyncd.status",

    -- 【关键】断线重连机制
    -- 如果设置为 true，当 Server B 网络不可达时，lsyncd 不会退出，
    -- 而是会不断重试，直到网络恢复。这对长时间训练非常重要。
    insist = true, -- 网络断开后会不断重试
}

-- 同步output目录下的输出文件
sync {
    -- 使用 rsync 命令模式（最稳健的模式）
    default.rsyncssh,

    -- 【源目录】：Server A 上正在训练输出的目录
    source = "/root/autodl-tmp/HMGS/output", -- 你的训练输出目录

    -- 【目标地址】：Server B 的 用户@IP
    host = "lzb@43.142.58.25",
    targetdir = "/mnt/home/lzb/projects/GSDF/output", -- Server B 的接收目录

    -- 【SSH 配置】
    -- rsyncssh 模式下，ssh 的参数需要放在 ssh 表中
    ssh = {
        port = 9997
    },

    -- 【核心安全配置：禁止删除】
    -- 默认为 true。如果设为 true，当你训练完删除了 A 的 xxx 目录，
    -- lsyncd 会忠实地把 B 上的备份也删掉！这违背了你的初衷。
    -- 设为 false 后，A 删文件，B 不会删；A 加文件，B 会加。只增不减。
    delete = false,
    
    -- rsync 的具体参数配置
    rsync = {
        -- 归档模式，保留时间戳、权限、所有者等信息
        archive = true,

        -- 压缩传输，节省带宽
        compress = true,

        -- 【关键】增量传输
        -- 确保 rsync 使用 delta 算法。对于 2GB 的 event 文件，
        -- 如果只追加了 10KB，它只会传输这 10KB 和少量的校验数据。
        whole_file = false, -- 允许增量传输

        -- 额外参数
        -- --partial: 支持断点续传。如果传输一半断网，下次从中断处继续，而不是重头传。
        -- --bwlimit=5000: (可选) 限制带宽为 5MB/s，防止占满带宽影响训练下载数据集。
        _extra = {"--partial"} -- 支持断点续传
    },

    -- 排除不需要同步的临时文件
    -- excludeFrom = "/path/to/excludes.txt",

    -- 【关键参数：延迟聚合】
    -- 延迟 5 秒再同步，避免文件正在写入时频繁触发
    -- 设置为 15 秒。这意味着文件发生变化后，lsyncd 会等 15 秒。
    -- 如果这 15 秒内又有新日志写入，它会合并在一起传输。
    -- 针对 TensorBoard 这种频繁写入的场景，设置 5-15 秒能极大降低系统负载，避免每秒都在建立 SSH 连接
    delay = 5
}

-- 同步runs目录下的日志文件
sync {
    -- 使用 rsync 命令模式（最稳健的模式）
    default.rsyncssh,

    -- 【源目录】：Server A 上正在训练输出的目录
    source = "/root/autodl-tmp/HMGS/runs", -- 你的训练输出目录

    -- 【目标地址】：Server B 的 用户@IP
    host = "lzb@43.142.58.25",
    targetdir = "/mnt/home/lzb/projects/GSDF/runs", -- Server B 的接收目录

    -- 【SSH 配置】
    -- rsyncssh 模式下，ssh 的参数需要放在 ssh 表中
    ssh = {
        port = 9997
    },

    -- 【核心安全配置：禁止删除】
    delete = false,
    
    rsync = {
        -- 归档模式，保留时间戳、权限、所有者等信息
        archive = true,

        -- 压缩传输，节省带宽
        compress = true,

        whole_file = false, -- 允许增量传输

        _extra = {"--partial"} -- 支持断点续传
    },

      delay = 5
}

-- 同步exp目录下的日志文件
sync {
    -- 使用 rsync 命令模式（最稳健的模式）
    default.rsyncssh,

    -- 【源目录】：Server A 上正在训练输出的目录
    source = "/root/autodl-tmp/HMGS/exp", -- 你的训练输出目录

    -- 【目标地址】：Server B 的 用户@IP
    host = "lzb@43.142.58.25",
    targetdir = "/mnt/home/lzb/projects/GSDF/exp", -- Server B 的接收目录
    -- 【SSH 配置】
    -- rsyncssh 模式下，ssh 的参数需要放在 ssh 表中
    ssh = {
        port = 9997
    },

    -- 【核心安全配置：禁止删除】
    delete = false,
    
    rsync = {
        -- 归档模式，保留时间戳、权限、所有者等信息
        archive = true,

        -- 压缩传输，节省带宽
        compress = true,

        whole_file = false, -- 允许增量传输

        _extra = {"--partial"} -- 支持断点续传
    },

      delay = 5
}