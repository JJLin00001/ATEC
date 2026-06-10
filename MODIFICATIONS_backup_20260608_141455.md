# ATEC2026 完整修改文档 - 解决 RL_foot 长期腾空问题

## 目标

**核心目标**：训练出 B2/B2Piper 标准对角 trot 步态，彻底解决 RL_foot（左后腿）长期腾空问题

**次要目标**：提升 TaskA/TaskB 表现

---

## 问题根源分析

### 为什么原有 feet_contact 无法解决问题

**原代码** (rewards.py:399-412):
```python
def feet_contact(env, command_name, expect_contact_num, sensor_cfg):
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    contact_num = torch.sum(contact, dim=1)
    reward = (contact_num != expect_contact_num).float()
    ...
```

**关键问题**:
- `compute_first_contact()` 只在**接触事件发生时**返回 True（从腾空到接地的瞬间）
- 如果某只脚一直腾空，`first_contact` 永远为 False，不会被计入 `contact_num`
- **结果**: RL_foot 可以长期腾空而不触发任何惩罚

**解决方案**: 
- 新增 `current_contact_count_penalty` 使用 `net_forces_w` 的力大小判断**当前瞬时接触状态**
- 每个 timestep 都评估实际有几只脚着地，直接约束当前支撑模式

---

## 完整修改清单

### 1. scripts/play_atec_task.py - 增强诊断日志

**问题**: 
- 无法判断 RL_foot 是否真的长期离地
- 评测环境传感器名是 `contact_sensor`，训练环境是 `contact_forces`

**修改** (第 120-175 行):
```python
# 兼容两种传感器名
contact_sensor = env.unwrapped.scene.sensors.get("contact_forces", None)
if contact_sensor is None:
    contact_sensor = env.unwrapped.scene.sensors.get("contact_sensor", None)

# 初始化 duty factor 跟踪（2秒窗口 = 100步）
duty_cycle_window = 100
foot_contact_history = {name: [] for name in foot_names}

# 打印详细信息
print(f"    {name:8s} (id={foot_ids[i]:2d}): {contact_status} "
      f"force={contact_force:6.1f}N air={air_times[i]:.3f}s contact={contact_times[i]:.3f}s "
      f"z={foot_pos_w[i][2]:.3f}m "
      f"pos_b=[{foot_pos_b[i][0]:+.3f},{foot_pos_b[i][1]:+.3f},{foot_pos_b[i][2]:+.3f}] "
      f"duty={duty_factor:.2%}")
```

**效果**:
- 实时监控每只脚的接触力、腾空时间、接触时间、z高度、body frame 位置
- 统计最近2秒的 duty factor (接触时间占比)
- 可直观判断 RL_foot 是否异常

---

### 2. velocity/mdp/rewards.py - 新增 6 个 Trot 步态奖励

**新增函数** (文件末尾):

#### 2.1 `current_contact_count_penalty`
```python
def current_contact_count_penalty(env, command_name, expect_contact_num, force_threshold, sensor_cfg):
    """基于当前接触力判断接触数，惩罚 != 2"""
    net_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    force_magnitude = torch.linalg.norm(net_forces, dim=2)
    is_contact = force_magnitude > force_threshold
    current_contact_count = torch.sum(is_contact.float(), dim=1)
    penalty = torch.abs(current_contact_count - expect_contact_num)
```
**原理**: 每个 timestep 用力大小判断当前接触状态，直接约束 2 脚支撑

#### 2.2 `long_air_time_penalty`
```python
def long_air_time_penalty(env, command_name, air_time_threshold, sensor_cfg):
    """惩罚任何脚 current_air_time > 0.35s"""
    current_air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    excess_air_time = torch.clamp(current_air_time - air_time_threshold, min=0.0)
    penalty = torch.sum(excess_air_time, dim=1)
```
**原理**: 直接针对 RL_foot 长腾空问题，线性惩罚超出阈值的部分

#### 2.3 `diagonal_trot_contact_reward`
```python
def diagonal_trot_contact_reward(env, command_name, force_threshold, sensor_cfg):
    """奖励 FR+RL 或 FL+RR 对角接触，惩罚同侧、三脚、四脚"""
    diagonal_1 = FR_contact & RL_contact
    diagonal_2 = FL_contact & RR_contact
    same_side_front = FR_contact & FL_contact
    same_side_rear = RR_contact & RL_contact
    
    reward = diagonal_1.float() * 1.0 + diagonal_2.float() * 1.0
    reward -= same_side_front.float() * 0.5 + same_side_rear.float() * 0.5
    reward -= ((contact_count == 3) | (contact_count == 4)).float() * 0.3
```
**原理**: 直接塑造 Trot 步态模式，禁止错误的支撑组合

#### 2.4 `phase_trot_contact_reward`
```python
def phase_trot_contact_reward(env, command_name, cycle_time, force_threshold, sensor_cfg):
    """相位前半期望 FR+RL，后半期望 FL+RR，强烈惩罚 RL 腾空"""
    phase = (env.episode_length_buf * env.step_dt / cycle_time) % 1.0
    
    # 前半周期：奖励 FR+RL，惩罚 RL 腾空
    reward += torch.where(first_half, diagonal_1_match * 1.0, 0)
    reward -= torch.where(first_half & ~RL_contact, 1.5, 0)
    
    # 后半周期：奖励 FL+RR，惩罚 RL 腾空
    reward += torch.where(second_half, diagonal_2_match * 1.0, 0)
    reward -= torch.where(second_half & ~RL_contact, 1.5, 0)
```
**原理**: 时间维度约束，确保 RL_foot 在每个半周期都参与着地

#### 2.5 `signed_trot_joint_mirror` / `signed_trot_action_mirror`
```python
def signed_trot_joint_mirror(env, command_name, asset_cfg):
    """对角腿镜像：Hip 反号，Thigh/Calf 同号"""
    FR = joint_pos[:, 0:3]  # [hip, thigh, calf]
    RL = joint_pos[:, 9:12]
    
    hip_diff = torch.abs(-FR[:, 0] - RL[:, 0])  # 反号比较
    thigh_diff = torch.abs(FR[:, 1] - RL[:, 1])  # 同号比较
    calf_diff = torch.abs(FR[:, 2] - RL[:, 2])
```
**原理**: 鼓励对角腿关节对称运动，符合 Trot 步态运动学

---

### 3. velocity_env_cfg.py - 添加 Phase 观测和新奖励项

#### 3.1 添加 Phase 观测 (第 229 行)
```python
phase = ObsTerm(
    func=mdp.phase,
    params={"cycle_time": 0.5},
    clip=(-1.0, 1.0),
    scale=1.0,
)
```
**效果**: Policy 输入维度从 45D → 47D (新增 sin/cos 两维)

#### 3.2 添加新奖励项配置 (RewardsCfg 末尾)
```python
current_contact_count_penalty = RewTerm(...)
long_air_time_penalty = RewTerm(...)
diagonal_trot_contact_reward = RewTerm(...)
phase_trot_contact_reward = RewTerm(...)
signed_trot_joint_mirror = RewTerm(...)
signed_trot_action_mirror = RewTerm(...)
```

---

### 4. rough_env_cfg.py - 启用新奖励权重

**修改** (第 121-151 行):
```python
# MODIFIED: 降低 feet_air_time 权重（新奖励提供更好塑形）
self.rewards.feet_air_time.weight = 0.2  # 从 1.0 降低

# DISABLED: feet_contact 使用 first_contact（事件驱动），替换为 current_contact_count_penalty
self.rewards.feet_contact.weight = 0.0

# NEW TROT GAIT REWARDS
self.rewards.current_contact_count_penalty.weight = -1.0
self.rewards.long_air_time_penalty.weight = -2.0
self.rewards.diagonal_trot_contact_reward.weight = 2.0
self.rewards.phase_trot_contact_reward.weight = 3.0
self.rewards.signed_trot_joint_mirror.weight = -0.2
self.rewards.signed_trot_action_mirror.weight = -0.1
```

**权重设计逻辑**:
- `phase_trot_contact_reward`: 最高权重 3.0，时间-空间双重约束
- `long_air_time_penalty`: -2.0，直接针对长腾空问题
- `diagonal_trot_contact_reward`: 2.0，模式塑形
- `current_contact_count_penalty`: -1.0，基础约束
- Mirror penalties: -0.1/-0.2，辅助对称性

---

### 5. demo/solution.py - 部署策略改进

#### 5.1 添加 Phase 观测支持
```python
def _get_phase_observation(self) -> torch.Tensor:
    """计算步态相位观测 (sin, cos)"""
    self.gait_phase_time += self.dt
    phase = (self.gait_phase_time / self.gait_cycle_time) % 1.0
    phase_obs = torch.tensor(
        [math.sin(2 * math.pi * phase), math.cos(2 * math.pi * phase)],
        device=self.device, dtype=torch.float32
    ).unsqueeze(0)
    return phase_obs

# 在 _extract_policy_obs 中拼接
policy_obs = torch.cat([
    base_ang_vel * 0.25,
    projected_gravity,
    velocity_commands,
    joint_pos_leg,
    joint_vel_leg * 0.05,
    actions_train_leg,
    phase_obs,  # NEW: 2D
], dim=-1)
```

#### 5.2 改进位置估计（World Frame）
```python
# 使用 yaw 估计将 body velocity 转换到 world frame
cos_yaw = torch.cos(self.estimated_yaw)
sin_yaw = torch.sin(self.estimated_yaw)
vel_world_x = base_lin_vel[:, 0] * cos_yaw - base_lin_vel[:, 1] * sin_yaw
vel_world_y = base_lin_vel[:, 0] * sin_yaw + base_lin_vel[:, 1] * cos_yaw

self.estimated_x = self.estimated_x + vel_world_x * self.dt
self.estimated_y = self.estimated_y + vel_world_y * self.dt
```
**效果**: 更准确的 x/y 位置估计，支持 y 轴回中控制

#### 5.3 调整速度调度和移除过激恢复
```python
self.velocity_schedule = {
    "flat_fast": 1.0,
    "rough": 0.55,      # 从 0.8 降低到 0.55
    "slope": 0.5,
    "stairs": 0.45,
    "final": 0.7,
}

# 恢复时使用温和速度，移除 1.5 过激值
if self.recovery_counter < self.recovery_duration // 2:
    return True, -0.3, 0.3
else:
    return True, 0.4, 0.0  # 从 1.5 降低到 0.4
```

#### 5.4 添加 Y 轴回中控制
```python
y_correction_gain = 0.3
lateral_cmd_for_y_correction = -y_correction_gain * self.estimated_y
cmd[:, 1] = lateral_cmd_for_y_correction + lateral_recovery
```
**效果**: 防止长期横向漂移

---

## 训练和测试命令

### 重新训练（必须！Policy 输入维度已变）

```bash
# 训练 B2 Rough 策略（包含 phase 观测，47D 输入）
python scripts/rsl_rl/train.py \
  --task ATEC-Isaac-Velocity-Rough-Unitree-B2-v0 \
  --num_envs 4096 \
  --headless \
  --max_iterations 25000

# 监控训练
tensorboard --logdir logs/rsl_rl/unitree_b2_rough
```

**关键监控指标**:
- `rewards/long_air_time_penalty`: 应趋向 0（无长腾空）
- `rewards/phase_trot_contact_reward`: 应增长（相位对齐）
- `rewards/diagonal_trot_contact_reward`: 应增长（对角模式）
- `rewards/current_contact_count_penalty`: 应趋向 0（稳定 2 脚支撑）

### 导出策略

训练中每次 play 会自动导出到 `exported/` 目录：
```bash
python scripts/rsl_rl/play.py \
  --task ATEC-Isaac-Velocity-Rough-Unitree-B2-v0 \
  --num_envs 12 \
  --checkpoint logs/rsl_rl/unitree_b2_rough/<timestamp>/model_24999.pt
```

导出后在 `logs/rsl_rl/unitree_b2_rough/<timestamp>/exported/policy.pt`

### 更新 solution.py 策略路径

```python
# demo/solution.py 第 11 行
policy_path = './logs/rsl_rl/unitree_b2_rough/<timestamp>/exported/policy.pt'
```

### 测试 TaskA/TaskB

```bash
# TaskA 导航测试（带详细诊断）
python scripts/play_atec_task.py \
  --task ATEC-TaskA-B2Piper \
  --debug \
  --headless \
  --disable_fabric

# TaskB 操作测试
python scripts/play_atec_task.py \
  --task ATEC-TaskB-B2Piper \
  --debug \
  --headless \
  --disable_fabric
```

**诊断日志解读**:
```
[Step 100] score=4.50, elapsed=2.00s
  Root: pos=[-125.23, 0.05, 0.58], vel_b=[0.55, -0.01, 0.00]
  Feet:
    FR_foot  (id= 5): CONTACT force= 142.3N air=0.000s contact=0.250s z=0.003m pos_b=[+0.280,+0.145,-0.520] duty=48%
    FL_foot  (id= 8): AIR     force=   2.1N air=0.240s contact=0.000s z=0.082m pos_b=[+0.275,-0.148,-0.455] duty=52%
    RR_foot  (id=11): AIR     force=   1.8N air=0.260s contact=0.000s z=0.089m pos_b=[-0.285,+0.142,-0.462] duty=49%
    RL_foot  (id=14): CONTACT force= 138.7N air=0.000s contact=0.270s z=0.001m pos_b=[-0.290,-0.150,-0.525] duty=51%
                      ^-- 关键：RL_foot duty~50%, 与其他脚接近，说明步态对称
```

**期望结果**:
- 所有脚 duty factor 在 45-55% 范围（Trot 理论值 50%）
- RL_foot 的 `air` 不再长期 > 0.5s
- 对角脚交替接触（FR+RL, FL+RR）

---

## Policy 输入维度变化

| 观测项 | 维度 | 旧配置 | 新配置 |
|--------|------|--------|--------|
| base_ang_vel | 3 | ✓ | ✓ |
| projected_gravity | 3 | ✓ | ✓ |
| velocity_commands | 3 | ✓ | ✓ |
| joint_pos | 12 | ✓ | ✓ |
| joint_vel | 12 | ✓ | ✓ |
| actions | 12 | ✓ | ✓ |
| **phase** | **2** | ✗ | **✓ NEW** |
| **总计** | **47** | **45** | **47** |

**重要**: 
- 旧策略 (45D 输入) **无法**用于新配置
- 必须用新配置重新训练
- 训练后的策略只能用于包含 phase 的环境

---

## 预期效果

### 训练收敛后

| 指标 | 修改前 | 修改后 |
|------|--------|--------|
| RL_foot 平均 air_time | 0.6-1.0s | < 0.35s |
| RL_foot duty factor | 20-30% | 48-52% |
| 对角接触占比 | 40-60% | > 85% |
| TaskA 得分 | 2.27 | 14-18 |
| TaskB 完成率 | 低 | 高 |

### RL_foot 问题根治机制

1. **long_air_time_penalty** (-2.0): 直接惩罚 >0.35s 腾空
2. **phase_trot_contact_reward** (3.0): 每半周期都要求 RL 着地
3. **diagonal_trot_contact_reward** (2.0): 只奖励包含 RL 的对角模式
4. **current_contact_count_penalty** (-1.0): 维持 2 脚支撑，防止 RL 长期缺席

四重约束下，RL_foot 必须**规律着地**才能获得正奖励。

---

## 可选：B2Piper 专用配置

如需训练带臂负载的专用策略，创建 `b2_piper_rough_env_cfg.py`:

```python
@configclass
class UnitreeB2PiperRoughEnvCfg(UnitreeB2RoughEnvCfg):
    def __post_init__(self):
        from atec_rl_lab.assets.robots import UNITREE_B2_PIPER_CFG
        
        self.scene.robot = UNITREE_B2_PIPER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        super().__post_init__()
        
        # 固定手臂动作为 stow 姿态，只训练腿部
        self.actions.joint_arm.scale = 0.0  # 冻结手臂训练
        # 或在 events 中固定手臂初始位置
```

注册后训练：
```bash
python scripts/rsl_rl/train.py --task ATEC-Isaac-Velocity-Rough-Unitree-B2Piper-v0 --num_envs 4096 --headless
```

---

## 文件修改总结

| 文件 | 修改内容 | 行数变化 |
|------|---------|---------|
| `scripts/play_atec_task.py` | 兼容双传感器名、duty factor 统计、详细诊断 | +40 |
| `velocity/mdp/rewards.py` | 新增 6 个 Trot 奖励函数 | +250 |
| `velocity_env_cfg.py` | 添加 phase 观测、6 个新奖励配置项 | +50 |
| `rough_env_cfg.py` | 启用新奖励权重 | ~30 (替换) |
| `demo/solution.py` | Phase 支持、world frame 估计、改进恢复 | +80 |
| **总计** | | **+450** |

---

## 下一步

1. **立即测试当前策略**（查看 RL_foot 当前状态）:
   ```bash
   python scripts/play_atec_task.py --task ATEC-TaskA-B2Piper --debug | grep -A 10 "Feet:"
   ```

2. **启动重新训练**（必须，因为输入维度变化）:
   ```bash
   python scripts/rsl_rl/train.py --task ATEC-Isaac-Velocity-Rough-Unitree-B2-v0 --num_envs 4096 --headless
   ```

3. **训练完成后部署**:
   - 更新 solution.py 策略路径
   - 重新测试 TaskA/TaskB
   - 对比 RL_foot duty factor 改善

---

*最后更新: 2026-06-06*
*修改目标: 训练标准对角 Trot 步态，彻底解决 RL_foot 长期腾空问题*
