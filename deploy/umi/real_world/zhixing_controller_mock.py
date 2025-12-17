# zhixing_controller_mock.py
"""
Mock 版本 ZhixingController
- 不打开任何串口
- 不依赖 ZhixingDriver
- 内部用简单线性模型模拟 pos/width 变化
- 所有共享内存/队列行为与真机一致
"""
import os
import time
import numpy as np
import enum
import multiprocessing as mp
from queue import Queue
from multiprocessing.managers import SharedMemoryManager
from deploy.umi.shared_memory.shared_memory_queue import (
    SharedMemoryQueue, Empty)
from deploy.umi.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from deploy.umi.common.precise_sleep import precise_wait

# 复用原文件里的 Command 定义
class Command(enum.Enum):
    SHUTDOWN = 0
    SCHEDULE_WAYPOINT = 1
    RESTART_PUT = 2


class ZhixingController(mp.Process):
    """
    与 zhixing_controller.ZhixingController 完全一致的公开接口。
    内部用假数据模拟夹爪运动。
    """
    def __init__(self,
                 shm_manager: SharedMemoryManager,
                 serial="MOCK_SERIAL",
                 baud=115200,
                 frequency=15,
                 get_max_k=None,
                 command_queue_size=1024,
                 launch_timeout=3,
                 receive_latency=0.0,
                 open_width=0.12,          # unit: m
                 closed_width=0.0,         # unit: m
                 force=30,
                 verbose=False):
        super().__init__(name="ZhixingControllerMock")
        self.serial = serial
        self.serial_dev = "/dev/ttyMOCK"          # 假装找到的串口
        self.baud = baud
        self.frequency = frequency
        self.launch_timeout = launch_timeout
        self.receive_latency = receive_latency
        self.verbose = verbose

        self.open_width = open_width
        self.closed_width = closed_width
        self.force = force

        if get_max_k is None:
            get_max_k = int(frequency * 10)

        # 构造和原文件一模一样的共享内存队列/环缓冲
        example = {
            'cmd': Command.SCHEDULE_WAYPOINT.value,
            'target_pos': 0.0,
            'target_time': 0.0
        }
        input_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            buffer_size=command_queue_size
        )

        example = {
            'gripper_position': 0.0,
            'gripper_receive_timestamp': time.time(),
            'gripper_timestamp': time.time(),
            'gripper_reached': False
        }
        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=frequency
        )

        self.ready_event = mp.Event()
        self.input_queue = input_queue
        self.ring_buffer = ring_buffer
        self.waypoint_queue = Queue(maxsize=1000)

        # 内部状态
        self._curr_pos = 1.0          # 初始全开
        self._target_pos = 1.0
        self._move_speed = 2.0        # pos/s，假装移动速度

    # ---------------- 与原文件一致的 public API -----------------
    def width_to_pos(self, width):
        return ((width - self.closed_width) /
                (self.open_width - self.closed_width))

    def pos_to_width(self, pos):
        return (pos * (self.open_width - self.closed_width) +
                self.closed_width)

    def start(self, wait=True):
        super().start()
        if wait:
            self.start_wait()
        if self.verbose:
            print(f"[ZhixingControllerMock] Mock controller spawned at {self.pid}")

    def stop(self, wait=True):
        msg = {'cmd': Command.SHUTDOWN.value}
        self.input_queue.put(msg)
        if wait:
            self.stop_wait()

    def start_wait(self):
        self.ready_event.wait(self.launch_timeout)
        assert self.is_alive()

    def stop_wait(self):
        self.join()

    @property
    def is_ready(self):
        return self.ready_event.is_set()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def schedule_waypoint(self, pos: float, target_time: float):
        msg = {
            'cmd': Command.SCHEDULE_WAYPOINT.value,
            'target_pos': pos,
            'target_time': target_time
        }
        self.input_queue.put(msg)

    def restart_put(self, start_time):
        self.input_queue.put({
            'cmd': Command.RESTART_PUT.value,
            'target_time': start_time
        })

    def get_state(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k=k, out=out)

    def get_all_state(self):
        return self.ring_buffer.get_all()

    # ---------------- 进程主循环 -----------------
    def run(self):
        try:
            t_start = time.monotonic()
            iter_idx = 0
            keep_running = True
            dt = 1 / self.frequency

            # 初始发布一次状态
            self._publish_state()

            while keep_running:
                t_now = time.monotonic()

                # 1. 模拟运动：每周期朝 target 靠近一点
                err = self._target_pos - self._curr_pos
                step = np.clip(err, -self._move_speed * dt, self._move_speed * dt)
                self._curr_pos += step

                # 2. 发布当前状态
                self._publish_state()

                # 3. 处理指令队列
                try:
                    cmds = self.input_queue.get_all()
                    n_cmd = len(cmds['cmd'])
                except Empty:
                    n_cmd = 0

                for i in range(n_cmd):
                    command = {k: v[i] for k, v in cmds.items()}
                    cmd = command['cmd']

                    if cmd == Command.SHUTDOWN.value:
                        keep_running = False
                        break
                    elif cmd == Command.SCHEDULE_WAYPOINT.value:
                        width = command['target_pos']
                        pos = self.width_to_pos(width)
                        self._target_pos = float(pos)
                    elif cmd == Command.RESTART_PUT.value:
                        t_start = command['target_time'] - time.time() + time.monotonic()
                        iter_idx = 1

                # 4. 首次循环后标记 ready
                if iter_idx == 0:
                    self.ready_event.set()
                iter_idx += 1

                # 5. 精确休眠
                t_end = t_start + dt * iter_idx
                precise_wait(t_end=t_end, time_func=time.monotonic)

        finally:
            self.ready_event.set()
            if self.verbose:
                print("[ZhixingControllerMock] Mock controller exits.")

    # ---------------- 内部辅助 -----------------
    def _publish_state(self):
        state = {
            'gripper_position': self.pos_to_width(self._curr_pos),
            'gripper_receive_timestamp': time.time(),
            'gripper_timestamp': time.time() - self.receive_latency,
            'gripper_reached': abs(self._curr_pos - self._target_pos) < 1e-3
        }
        self.ring_buffer.put(state)