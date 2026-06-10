# ATEC2026 Challenge - Trot Gait Fix & Straight-Line Optimization

## Executive Summary

**Goal**: Train B2/B2Piper with coordinated diagonal trot gait, balanced duty cycles between diagonal pairs, and enable straight-line forward motion on TaskA.

**Problem Identified**:
- Previous score 3.11 ≈ max_x = -92.8m (failure in TaskA random_rough initial section)
- RL_foot (left rear) had long-term airborne issue (duty ~20-30%, should be ~50%)
- Diagonal pair FR+RL vs FL+RR duty imbalance caused asymmetric gait
- Lateral command correction broke four-legged diagonal gait coordination
- Phase reward had asymmetric logic (penalized RL in both half-cycles)

**Root Causes**:
1. **`feet_contact` reward failure**: Used `compute_first_contact()` which only triggers on air→ground EVENTS, cannot detect continuous airborne state
2. **Non-symmetric phase reward**: `phase_trot_contact_reward` penalized `~RL_contact` in BOTH first and second half-cycles, giving RL special (negative) treatment
3. **Foot order assumption**: Rewards used `".*_foot"` pattern without `preserve_order=True`, risking incorrect foot index mapping
4. **Lateral command interference**: Using `cmd[:,1]` for Y-drift correction caused lateral body motion that disrupted diagonal trot synchronization
5. **Training on B2 but deploying on B2Piper**: Different inertia/mass distribution due to arms

---

## Key Technical Changes

### 1. Fixed `phase_trot_contact_reward` Asymmetric Logic

**Before** (lines 820-873 in rewards.py):
```python
# First half: reward FR+RL diagonal, penalize if RL is airborne
reward -= torch.where(first_half & ~RL_contact, 1.5, 0)

# Second half: reward FL+RR diagonal, penalize if RL is airborne  
reward -= torch.where(second_half & ~RL_contact, 1.5, 0)  # BUG: RL penalized in BOTH halves!
```

**Problem**: RL_foot was penalized for being airborne in BOTH phase [0, 0.5) and [0.5, 1.0), while other feet only penalized in their expected half. This asymmetry caused RL to stay grounded excessively or be ignored entirely.

**After**:
```python
# First half: expect FR+RL
reward -= torch.where(first_half & ~FR_contact, 0.8, 0)  # Penalize missing FR
reward -= torch.where(first_half & ~RL_contact, 0.8, 0)  # Penalize missing RL
reward -= torch.where(first_half & FL_contact & RR_contact, 0.5, 0)  # Penalize FL+RR over-contact

# Second half: expect FL+RR
reward -= torch.where(second_half & ~FL_contact, 0.8, 0)  # Penalize missing FL
reward -= torch.where(second_half & ~RR_contact, 0.8, 0)  # Penalize missing RR
reward -= torch.where(second_half & FR_contact & RL_contact, 0.5, 0)  # Penalize FR+RL over-contact
```

**Why This Fixes Diagonal Pair Imbalance**: Now each foot has symmetric treatment:
- FR: penalized only in first half if missing, penalized in second half if over-contacting
- FL: penalized only in second half if missing, penalized in first half if over-contacting
- RR: same as FL
- RL: same as FR

No single foot gets double-penalty across both phases.

---

### 2. Explicit Foot Order with `preserve_order=True`

**Problem**: Using `sensor_cfg=SceneEntityCfg("contact_forces", body_names=".*_foot")` relies on regex match order, which may not guarantee `[FR, FL, RR, RL]` sequence.

**Solution**: All gait-related rewards now use:
```python
sensor_cfg = SceneEntityCfg("contact_forces", 
                           body_names=["FR_foot", "FL_foot", "RR_foot", "RL_foot"],
                           preserve_order=True)
```

**Affected rewards** (velocity_env_cfg.py + rough_env_cfg.py):
- `feet_air_time_variance`
- `current_contact_count_penalty`
- `long_air_time_penalty`
- `diagonal_trot_contact_reward`
- `phase_trot_contact_reward`
- `diagonal_pair_duty_balance_penalty` (new)

---

### 3. New Reward: `diagonal_pair_duty_balance_penalty`

**Purpose**: Penalize duty cycle imbalance between diagonal pairs (FR+RL vs FL+RR).

**Implementation** (rewards.py, appended):
```python
def diagonal_pair_duty_balance_penalty(env, command_name, sensor_cfg):
    """Penalize duty imbalance between FR+RL vs FL+RR diagonal pairs."""
    current_air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    current_contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    
    # FR, FL, RR, RL order
    pair1_air = (current_air_time[:, 0] + current_air_time[:, 3]) / 2.0  # FR+RL
    pair2_air = (current_air_time[:, 1] + current_air_time[:, 2]) / 2.0  # FL+RR
    
    pair1_contact = (current_contact_time[:, 0] + current_contact_time[:, 3]) / 2.0
    pair2_contact = (current_contact_time[:, 1] + current_contact_time[:, 2]) / 2.0
    
    penalty = torch.abs(pair1_air - pair2_air) + torch.abs(pair1_contact - pair2_contact)
    return penalty
```

**Why This Helps**: Previous rewards could converge to a solution where FR+RL pair has 60% duty and FL+RR pair has 40% duty (or vice versa), both satisfying phase sync but creating asymmetric gait. This reward directly penalizes such imbalance.

**Weight**: `-2.0` (rough_env_cfg.py line ~158)

---

### 4. Balanced Reward Weights

**Problem**: Previous weights (phase_trot=4.0, long_air_time=-3.0) were too rigid, causing the robot to prioritize gait constraints over velocity tracking and terrain adaptation.

**New Weights** (rough_env_cfg.py):
```python
# Velocity tracking - HIGHEST priority
self.rewards.track_lin_vel_xy_exp.weight = 6.0  # Was 3.0
self.rewards.track_ang_vel_z_exp.weight = 2.0   # Was 1.5

# Gait enforcement - balanced to avoid rigidity
self.rewards.phase_trot_contact_reward.weight = 1.5  # Reduced from 4.0
self.rewards.long_air_time_penalty.weight = -1.5     # Reduced from -3.0
self.rewards.diagonal_trot_contact_reward.weight = 2.5
self.rewards.current_contact_count_penalty.weight = -1.0
self.rewards.diagonal_pair_duty_balance_penalty.weight = -2.0  # NEW

# Disabled rewards that conflict
self.rewards.feet_air_time.weight = 0.0  # Can reward long air time
self.rewards.feet_contact.weight = 0.0   # Uses first_contact events, unreliable
```

**Rationale**:
- **Velocity tracking (6.0 + 2.0 = 8.0 total)** dominates to ensure forward progress
- **Gait constraints (1.5 + 2.5 + 2.0 = 6.0 positive, 1.5 + 1.0 + 2.0 = 4.5 negative)** provide shaping but don't overpower
- **feet_gait (3.0)** and **upward (4.0)** encourage dynamic motion


### 5. Pure-Pursuit Heading Control (solution.py)

**Problem**: Previous approach used lateral command `cmd[:,1] = -0.3 * estimated_y` to correct Y-drift. This caused:
- Lateral body motion that disrupted diagonal trot synchronization
- One diagonal pair compensating for lateral force, breaking duty balance

**Solution**: Use pure-pursuit heading controller that corrects path via yaw rotation only:

**New Control Logic** (solution.py lines 37-48, 175-244):
```python
# Pure-pursuit parameters
self.k_heading = 0.8           # Heading error gain
self.k_y = 0.6                 # Y-error to heading gain
self.lookahead = 2.0           # Lookahead distance
self.k_damping = 0.3           # Yaw rate damping
self.yaw_bias = 0.0            # Systematic drift correction (tune to -0.03~-0.08 if needed)

# In _get_velocity_commands():
desired_heading = atan2(-k_y * estimated_y, lookahead) + yaw_bias
heading_error = wrap(desired_heading - estimated_yaw)
yaw_rate_cmd = k_heading * heading_error - k_damping * yaw_rate_measured
yaw_rate_cmd = clip(yaw_rate_cmd, -0.5, 0.5)  # Gentle turning

cmd[:, 0] = forward_vel
cmd[:, 1] = clip(lateral_recovery, -0.08, 0.08)  # Minimal lateral, recovery only
cmd[:, 2] = yaw_rate_cmd
```

**Why This Preserves Trot Gait**: Yaw rotation keeps body motion purely forward-backward, allowing diagonal pairs to maintain synchronized duty cycles without lateral compensation.

---

### 6. Fixed Phase Generation Timing (solution.py)

**Problem**: Original code incremented `gait_phase_time` BEFORE computing phase:
```python
self.gait_phase_time += self.dt  # Increment first
phase = (self.gait_phase_time / self.gait_cycle_time) % 1.0  # Then compute
```
This caused a fixed one-timestep phase offset between training (which uses `episode_length_buf * step_dt`) and deployment.

**Solution** (solution.py lines 246-262):
```python
def _get_phase_observation(self):
    # Compute phase from CURRENT time (before increment)
    phase = ((self.gait_phase_time + self.phase_offset * self.gait_cycle_time) 
             / self.gait_cycle_time) % 1.0
    
    phase_obs = torch.tensor(
        [math.sin(2 * math.pi * phase), math.cos(2 * math.pi * phase)],
        device=self.device, dtype=torch.float32
    ).unsqueeze(0)
    
    # Increment time AFTER computing phase
    self.gait_phase_time += self.dt
    return phase_obs
```

**Added Parameter**:
```python
self.phase_offset = 0.0  # Tune: try 0.0, 0.125, 0.25, 0.375 for best alignment
```

---

### 7. UnitreeB2PiperRoughEnvCfg for Matched Training

**Problem**: Training on bare B2 (no arms) but deploying on B2Piper (with arms) causes:
- Different center of mass
- Different rotational inertia
- Different contact dynamics (arm mass affects body pitch)

**Solution**: New training config using full B2Piper model:

**File**: `source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/config/quadruped/unitree_b2_piper/rough_env_cfg.py`

**Key Points**:
```python
@configclass
class UnitreeB2PiperRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    leg_joint_names = [
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    ]
    
    def __post_init__(self):
        super().__post_init__()
        
        # Use B2Piper model
        self.scene.robot = UNITREE_B2_PIPER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        
        # Fix arm joints in stow configuration
        self.scene.robot.init_state.joint_pos.update({
            "arm_joint1": 0.0, "arm_joint2": 1.2, "arm_joint3": 0.0, "arm_joint4": -1.5,
            "arm_joint5": 0.0, "arm_joint6": 0.0, "arm_joint7": 0.0, "arm_joint8": 0.0,
        })
        
        # Policy only observes/controls leg joints (12 DOF)
        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = self.leg_joint_names
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = self.leg_joint_names
        self.actions.joint_pos.joint_names = self.leg_joint_names
        
        # Copy all reward weights from UnitreeB2RoughEnvCfg
        # ... (identical weights)
```

**Registered as**: `ATEC-Isaac-Velocity-Rough-Unitree-B2Piper-v0`

---

## Training Commands

### Option 1: Train on B2 (bare legs, faster)
```bash
python scripts/rsl_rl/train.py \
  --task ATEC-Isaac-Velocity-Rough-Unitree-B2-v0 \
  --num_envs 4096 \
  --headless \
  --max_iterations 25000
```

### Option 2: Train on B2Piper (with arms, matched dynamics) **RECOMMENDED**
```bash
python scripts/rsl_rl/train.py \
  --task ATEC-Isaac-Velocity-Rough-Unitree-B2Piper-v0 \
  --num_envs 4096 \
  --headless \
  --max_iterations 25000
```

**Monitor training**:
```bash
tensorboard --logdir logs/rsl_rl
```

**Key metrics to watch**:
- `rewards/long_air_time_penalty`: should approach 0 (no excessive air time)
- `rewards/phase_trot_contact_reward`: should increase (better phase sync)
- `rewards/diagonal_pair_duty_balance_penalty`: should approach 0 (balanced pairs)
- `rewards/track_lin_vel_xy_exp`: should remain high (forward progress maintained)

---

## Export and Deployment

### 1. Export trained policy
```bash
python scripts/rsl_rl/play.py \
  --task ATEC-Isaac-Velocity-Rough-Unitree-B2Piper-v0 \
  --num_envs 12 \
  --checkpoint logs/rsl_rl/unitree_b2_piper_rough/<timestamp>/model_24999.pt
```

This automatically exports to `logs/rsl_rl/unitree_b2_piper_rough/<timestamp>/exported/policy.pt`

### 2. Update solution.py
Edit line 11 in `demo/solution.py`:
```python
policy_path = './logs/rsl_rl/unitree_b2_piper_rough/<timestamp>/exported/policy.pt'
```

### 3. Test with enhanced diagnostics
```bash
# TaskA test (with detailed foot diagnostics)
python scripts/play_atec_task.py --task ATEC-TaskA-B2Piper --debug --headless --disable_fabric

# TaskB test
python scripts/play_atec_task.py --task ATEC-TaskB-B2Piper --debug --headless --disable_fabric
```

**Diagnostic output interpretation** (from play_atec_task.py `--debug`):
```
Feet:
    FR_foot  (id= 4): ▓ force= 120.5N air=0.120s contact=0.380s z=0.053m pos_b=[+0.285,+0.145,-0.470] duty=76.00%
    FL_foot  (id= 7): ░ force=   2.1N air=0.380s contact=0.120s z=0.182m pos_b=[+0.285,-0.145,-0.387] duty=24.00%
    RR_foot  (id=10): ░ force=   3.5N air=0.375s contact=0.125s z=0.175m pos_b=[-0.285,+0.145,-0.392] duty=25.00%
    RL_foot  (id=13): ▓ force= 118.2N air=0.125s contact=0.375s z=0.055m pos_b=[-0.285,-0.145,-0.468] duty=75.00%
```

**Good trot indicators**:
- FR+RL duty ≈ 48-52% each, FL+RR duty ≈ 48-52% each
- Diagonal pairs alternate: when FR+RL show `▓` (contact), FL+RR show `░` (air)
- Pair duty balance: `|avg(FR,RL) - avg(FL,RR)| < 5%`

**Bad gait indicators**:
- Any foot with duty < 35% or > 65% (asymmetric load)
- FR+RL avg duty = 60%, FL+RR avg duty = 40% → pair imbalance
- RL duty = 20% → still airborne too long (reward weights may need tuning)


## Tuning Parameters (After Initial Training)

If testing shows systematic drift or phase misalignment:

### 1. Yaw bias (solution.py line 45)
```python
self.yaw_bias = 0.0  # Default
```
**If robot drifts left consistently**: try `-0.03` to `-0.08`  
**If robot drifts right consistently**: try `+0.03` to `+0.08`

### 2. Phase offset (solution.py line 69)
```python
self.phase_offset = 0.0  # Default
```
**If gait looks "late" (feet contact after expected phase)**: try `0.125` or `0.25`  
**If gait looks "early"**: try `-0.125` or `0.375` (equivalent to -0.125)

### 3. Reward weights (rough_env_cfg.py)
**If gait is too rigid on rough terrain**:
- Reduce `phase_trot_contact_reward.weight` from 1.5 to 1.0
- Reduce `long_air_time_penalty.weight` from -1.5 to -1.0

**If diagonal pair imbalance persists**:
- Increase `diagonal_pair_duty_balance_penalty.weight` from -2.0 to -3.0

**If robot still struggles with forward velocity**:
- Increase `track_lin_vel_xy_exp.weight` from 6.0 to 8.0

---

## Expected Results

### Baseline (Before Modifications)
| Metric | Value |
|--------|-------|
| TaskA Score | 3.11 |
| Max X Position | ~-92.8m |
| Failure Point | random_rough initial section |
| RL_foot duty | 20-30% |
| Diagonal pair balance | FR+RL: 60%, FL+RR: 40% |

### Target (After Modifications)
| Metric | Target |
|--------|--------|
| TaskA Score | 14-18 |
| Max X Position | > 100m |
| RL_foot duty | 48-52% |
| Diagonal pair balance | FR+RL: 50%, FL+RR: 50% (±3%) |
| Diagonal contact % | > 85% |
| Y-drift at x=100m | < 2m |

---

## Technical Deep-Dive: Why Lateral Command Breaks Diagonal Trot

### Force Analysis

In trot gait, diagonal pairs (FR+RL, FL+RR) alternate support. When a lateral command `cmd[:,1] = -0.3 * y` is issued:

1. **Body applies lateral force** to match command
2. **Ground reaction force must balance** this lateral component
3. **Asymmetric leg loading**: 
   - Left legs (FL+RL) push harder medially
   - Right legs (FR+RR) push harder laterally
4. **Duty cycle compensation**:
   - The pair aligned with lateral force direction spends more time in contact (higher duty)
   - The opposite pair reduces contact time (lower duty)

### Example Scenario
```
Y-drift: +1.0m (right of centerline)
Lateral command: -0.3 m/s (push left)

Result:
- Left legs (FL+RL) must generate rightward ground reaction → increased contact time
- Right legs (FR+RR) can reduce contact time → lower duty
- FR+RL pair avg duty: 55%
- FL+RR pair avg duty: 45%
```

### Why Yaw Control Fixes This

Pure-pursuit heading control generates `cmd[:,2]` (yaw rate) only:
```
cmd[:,0] = forward_vel (always positive)
cmd[:,1] = 0 (or clip to ±0.08 for recovery only)
cmd[:,2] = yaw_rate (corrects heading toward centerline)
```

**Body motion remains forward-backward**, no sustained lateral forces. Diagonal pairs maintain natural 50-50 duty split.

---

## Files Modified

1. **source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/mdp/rewards.py**
   - Fixed `phase_trot_contact_reward` (lines 820-873)
   - Fixed `signed_trot_action_mirror` (line 931: use `env.action_manager.action`)
   - Added `diagonal_pair_duty_balance_penalty` (lines 970-1024)

2. **source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/velocity_env_cfg.py**
   - Updated `feet_air_time_variance` with explicit foot order (line 550)
   - Updated all new gait rewards with explicit foot order (lines 652-710)
   - Added `diagonal_pair_duty_balance_penalty` RewTerm (lines 712-718)

3. **source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/config/quadruped/unitree_b2/rough_env_cfg.py**
   - Updated all gait reward weights (lines 120-158)
   - Set all sensor_cfg with explicit foot order and `preserve_order=True`
   - Added `diagonal_pair_duty_balance_penalty` with weight=-2.0

4. **demo/solution.py**
   - Replaced heading control with pure-pursuit (lines 37-48)
   - Rewrote `_get_velocity_commands` (lines 175-244)
   - Fixed `_get_phase_observation` timing (lines 246-262)
   - Added `phase_offset` parameter (line 69)
   - Added `yaw_bias` parameter (line 45)

5. **source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/config/quadruped/unitree_b2_piper/** (NEW)
   - `rough_env_cfg.py`: Full B2Piper training config
   - `__init__.py`: Gym registration for `ATEC-Isaac-Velocity-Rough-Unitree-B2Piper-v0`

6. **source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/config/quadruped/__init__.py**
   - Added `from .unitree_b2_piper import *`

---

## Troubleshooting

### Issue: Robot still drifts off centerline
**Check**:
1. `estimated_y` in debug logs - is it accumulating drift?
2. Try increasing `k_y` from 0.6 to 0.8
3. Add small `yaw_bias` if drift is systematic

### Issue: Gait looks correct but RL_foot duty still low
**Check**:
1. Tensorboard: is `long_air_time_penalty` approaching 0?
2. If not, increase weight from -1.5 to -2.0
3. Check if `diagonal_pair_duty_balance_penalty` is active (should not be 0)

### Issue: Robot moves slowly or stops on rough terrain
**Check**:
1. Gait constraints may be too rigid
2. Reduce `phase_trot_contact_reward.weight` to 1.0
3. Increase `track_lin_vel_xy_exp.weight` to 8.0

### Issue: Training converges but diagonal pairs still imbalanced
**Check**:
1. Verify `preserve_order=True` in all sensor_cfg
2. Check if `diagonal_pair_duty_balance_penalty` is being called (add debug print)
3. Increase weight from -2.0 to -3.0

### Issue: Policy exported but solution.py crashes
**Check**:
1. Policy input dimension: must be 47D (includes 2D phase observation)
2. Verify `phase` observation exists in velocity_env_cfg.py PolicyCfg
3. Check policy path in solution.py line 11

---

## Conclusion

These modifications address the root causes of RL_foot long-term airborne issue and diagonal pair duty imbalance by:

1. **Symmetric phase reward logic** - no single foot gets special treatment
2. **Explicit foot ordering** - eliminates index mapping ambiguity
3. **Diagonal pair duty balance** - directly enforces FR+RL ≈ FL+RR duty
4. **Pure-pursuit path control** - preserves trot coordination without lateral interference
5. **Matched training dynamics** - B2Piper config ensures deployment consistency

The reward weight balance prioritizes forward velocity (6.0) while providing sufficient gait shaping (total ~6.0 positive, ~4.5 negative) to maintain coordinated diagonal trot without over-constraining rough terrain adaptation.

Expected outcome: TaskA score 14-18, straight-line motion with < 2m lateral drift at 100m, and balanced diagonal pair duty cycles (48-52% each).
