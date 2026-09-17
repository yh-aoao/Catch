# Roll XY Waiting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 roll Tracking 改为桌面阶段 XY 优先、桌边下方等待，降低手撞桌风险。

**Architecture:** 保留 `roll_waiting.target()` 的预测拦截接口，在 `roll_rewards` 增加只基于目标 XY 的等待塑形；环境 Tracking 奖励使用预测目标的 XY 和目标朝向，不再在桌面阶段追求真实球 Z。桌面高度保持不变，安全距离作为奖励惩罚而不是新的硬门槛。

**Tech Stack:** Python, NumPy, unittest, MuJoCo environment reward code.

---

### Task 1: Add XY waiting regression tests

**Files:**
- Modify: `catch_it_copy/tests/test_roll_rewards.py`

- [ ] **Step 1: Write tests**

  Add tests asserting that the waiting target keeps fixed Z when the ball Z changes, and that XY movement changes the shaping signal while Z-only movement does not.

- [ ] **Step 2: Run the focused tests**

  Run from `catch_it_copy`: `python tests/test_roll_rewards.py -k 'waiting or shaping'`.
  Expected: the new behavior test fails before implementation.

### Task 2: Implement XY-first waiting shaping

**Files:**
- Modify: `catch_it_copy/gym_dcmm/utils/roll_waiting.py`
- Modify: `catch_it_copy/gym_dcmm/envs/DcmmVecEnv.py`
- Modify: `catch_it_copy/configs/env/DcmmCfg.py`

- [ ] **Step 1: Add XY-only waiting shaping helper**

  Keep the current target prediction, but make the waiting branch reward only XY distance/progress and a bounded high-hand penalty. Do not add a Z distance reward while `waiting` is true.

- [ ] **Step 2: Use predicted target for waiting orientation**

  In roll Tracking, use the predicted target direction during waiting for palm/finger alignment; use the actual ball direction only after the ball leaves the table.

- [ ] **Step 3: Strengthen safety parameters**

  Add explicit waiting safety weights in `DcmmCfg.py` with conservative defaults, without changing table geometry or termination logic.

### Task 3: Verify roll behavior surface

**Files:**
- Test: `catch_it_copy/tests/test_roll_rewards.py`

- [ ] **Step 1: Run focused tests**

  Run `python tests/test_roll_rewards.py -k 'waiting or shaping'`.
  Expected: PASS.

- [ ] **Step 2: Run the full roll suite**

  Run `python tests/test_roll_rewards.py`.
  Expected: all tests PASS.

- [ ] **Step 3: Inspect the diff**

  Confirm only roll reward/target/config/tests/docs changed; do not alter bounce or basket behavior.