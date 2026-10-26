namespace supermimic_vibed {

using mimic_lite::CTX_FRAMES;
using mimic_lite::D_IL_OF_MJ;
using mimic_lite::F_POS;
using mimic_lite::F_Q;
using mimic_lite::F_QUAT;
using mimic_lite::FRAME;
using mimic_lite::frame_at;
using mimic_lite::MAX_FRAMES;
using mimic_lite::PLAN_FRAMES;
using mimic_lite::PLAN_SEED;
using mimic_lite::q_mul;
using mimic_lite::q_yaw;
using mimic_lite::QPOS;
using mimic_lite::REPLAN_TICKS;

constexpr int NUM_ACTIONS = 29;
constexpr int NUM_OWNED = 15;
constexpr int PROPRIO = 99;
constexpr int HIST = 10;
constexpr int OBS = PROPRIO * HIST;
constexpr int GOAL_K = 10;
constexpr int GOAL_FRAME = 73;
constexpr int GOAL = GOAL_K * GOAL_FRAME;
constexpr float CONTROL_DT = 0.02f;
constexpr float QD_GOAL_SCALE = 0.1f;

enum Rot6 { ROT6_ROWS = 0, ROT6_COLS = 1 };

#define SUPERMIMIC_VIBED_DEFAULTS                                          \
  -0.312f, 0.0f, 0.0f, 0.669f, -0.363f, 0.0f, -0.312f, 0.0f, 0.0f, 0.669f, \
      -0.363f, 0.0f, 0.0f, 0.0f, 0.0f, 0.2f, 0.2f, 0.0f, 0.6f, 0.0f, 0.0f, \
      0.0f, 0.2f, -0.2f, 0.0f, 0.6f, 0.0f, 0.0f, 0.0f

__device__ const float D_DEFAULTS[NUM_ACTIONS] = {SUPERMIMIC_VIBED_DEFAULTS};

constexpr float SCALE_7520_14 = 0.5475464463233948f;
constexpr float SCALE_7520_22 = 0.3506614565849304f;
constexpr float SCALE_5020 = 0.4385773241519928f;

__device__ const float D_SCALE[NUM_OWNED] = {
    SCALE_7520_14,
    SCALE_7520_22,
    SCALE_7520_14,
    SCALE_7520_22,
    SCALE_5020,
    SCALE_5020,
    SCALE_7520_14,
    SCALE_7520_22,
    SCALE_7520_14,
    SCALE_7520_22,
    SCALE_5020,
    SCALE_5020,
    SCALE_7520_14,
    SCALE_5020,
    SCALE_5020
};

constexpr float KP_7520_14 = 40.17923863450712f;
constexpr float KP_7520_22 = 99.09842777666111f;
constexpr float KP_5020 = 28.50124619574858f;
constexpr float KD_7520_14 = 2.557889775413375f;
constexpr float KD_7520_22 = 6.308801853496639f;
constexpr float KD_5020 = 1.814445686584846f;

const float KPS[NUM_OWNED] = {
    KP_7520_14,
    KP_7520_22,
    KP_7520_14,
    KP_7520_22,
    KP_5020,
    KP_5020,
    KP_7520_14,
    KP_7520_22,
    KP_7520_14,
    KP_7520_22,
    KP_5020,
    KP_5020,
    KP_7520_14,
    KP_5020,
    KP_5020
};
const float KDS[NUM_OWNED] = {
    KD_7520_14,
    KD_7520_22,
    KD_7520_14,
    KD_7520_22,
    KD_5020,
    KD_5020,
    KD_7520_14,
    KD_7520_22,
    KD_7520_14,
    KD_7520_22,
    KD_5020,
    KD_5020,
    KD_7520_14,
    KD_5020,
    KD_5020
};

__device__ inline void q_rotate(
    const float* q,
    const float* v,
    float* out
) {
  const float w = q[0], x = q[1], y = q[2], z = q[3];
  const float tx = 2.0f * (y * v[2] - z * v[1]);
  const float ty = 2.0f * (z * v[0] - x * v[2]);
  const float tz = 2.0f * (x * v[1] - y * v[0]);
  out[0] = v[0] + w * tx + (y * tz - z * ty);
  out[1] = v[1] + w * ty + (z * tx - x * tz);
  out[2] = v[2] + w * tz + (x * ty - y * tx);
}

__device__ inline void q_rotate_inv(
    const float* q,
    const float* v,
    float* out
) {
  const float c[4] = {q[0], -q[1], -q[2], -q[3]};
  q_rotate(c, v, out);
}

__device__ inline void q_log(
    const float* q,
    float* out
) {
  const float s = q[0] < 0.0f ? -1.0f : 1.0f;
  const float w = s * q[0];
  const float n = sqrtf(q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
  const float k = n > 1e-7f ? 2.0f * atan2f(n, w) / n : 2.0f;
  for (int i = 0; i < 3; ++i) out[i] = s * k * q[1 + i];
}

__device__ inline void rot6(
    const float* q,
    int layout,
    float* out
) {
  const float w = q[0], x = q[1], y = q[2], z = q[3];
  const float r00 = 1.0f - 2.0f * (y * y + z * z);
  const float r01 = 2.0f * (x * y - w * z);
  const float r10 = 2.0f * (x * y + w * z);
  const float r11 = 1.0f - 2.0f * (x * x + z * z);
  const float r20 = 2.0f * (x * z - w * y);
  const float r21 = 2.0f * (y * z + w * x);
  if (layout == ROT6_ROWS) {
    out[0] = r00, out[1] = r01, out[2] = r10;
    out[3] = r11, out[4] = r20, out[5] = r21;
  } else {
    out[0] = r00, out[1] = r10, out[2] = r20;
    out[3] = r01, out[4] = r11, out[5] = r21;
  }
}

__device__ inline int clamp_frame(
    int f,
    int n
) {
  const int last = n > 0 ? n - 1 : 0;
  return f < 0 ? 0 : (f > last ? last : f);
}

__global__ void k_supermimic_vibed_obs(
    const float* __restrict__ motion,
    const int* __restrict__ len,
    const int* __restrict__ cursor,
    const float* __restrict__ motor_q,
    const float* __restrict__ motor_dq,
    const float* __restrict__ gyro,
    const float* __restrict__ gravity,
    const float* __restrict__ base_quat,
    const float* __restrict__ yaw_off,
    const float* __restrict__ arm_pose,
    const float* __restrict__ prev_act,
    float* __restrict__ hist,
    float* __restrict__ goal,
    int layout,
    long long tick,
    int envs
) {
  const int env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= envs) return;

  const float* mq = motor_q + env * POLICY_NUM_MOTOR;
  const float* mdq = motor_dq + env * POLICY_NUM_MOTOR;
  const float* m = motion + size_t(env) * MAX_FRAMES * FRAME;
  const int n = len[env];
  const int now = clamp_frame(cursor[env], n);
  const float* f0 = frame_at(m, n, now);

  const float off[4] =
      {cosf(0.5f * yaw_off[env]), 0.0f, 0.0f, sinf(0.5f * yaw_off[env])};
  float aligned[4];
  q_mul(off, base_quat + env * 4, aligned);
  const float inv[4] = {aligned[0], -aligned[1], -aligned[2], -aligned[3]};

  float* h = hist + size_t(env) * OBS;
  if (tick > 0) {
    for (int i = 0; i < OBS - PROPRIO; ++i) h[i] = h[i + PROPRIO];
  }
  float* p = h + OBS - PROPRIO;
  for (int j = 0; j < NUM_ACTIONS; ++j) {
    p[j] = mq[j] - D_DEFAULTS[j];
    p[29 + j] = mdq[j];
    p[64 + j] = prev_act[env * NUM_ACTIONS + j];
  }
  for (int k = 0; k < 3; ++k) {
    p[58 + k] = gyro[env * 3 + k];
    p[61 + k] = gravity[env * 3 + k];
  }
  float rel[4];
  q_mul(inv, f0 + F_QUAT, rel);
  rot6(rel, layout, p + 93);
  if (tick == 0) {
    for (int i = 0; i < HIST - 1; ++i) {
      for (int k = 0; k < PROPRIO; ++k) h[i * PROPRIO + k] = p[k];
    }
  }

  const float yaw = q_yaw(f0 + F_QUAT);
  const float c = cosf(yaw), s = sinf(yaw);
  const float head_inv[4] = {cosf(0.5f * yaw), 0.0f, 0.0f, -sinf(0.5f * yaw)};

  float* o = goal + size_t(env) * GOAL;
  for (int k = 0; k < GOAL_K; ++k) {
    const int fi = clamp_frame(now + 1 + k, n);
    const int lo = clamp_frame(fi - 1, n), hi = clamp_frame(fi + 1, n);
    const float* fr = frame_at(m, n, fi);
    const float* a = frame_at(m, n, lo);
    const float* b = frame_at(m, n, hi);
    const float rate = hi > lo ? 1.0f / (float(hi - lo) * CONTROL_DT) : 0.0f;
    float* g = o + k * GOAL_FRAME;

    const float dx = fr[F_POS + 0] - f0[F_POS + 0];
    const float dy = fr[F_POS + 1] - f0[F_POS + 1];
    g[0] = c * dx + s * dy;
    g[1] = -s * dx + c * dy;
    g[2] = fr[F_POS + 2] - f0[F_POS + 2];

    float hq[4];
    q_mul(head_inv, fr + F_QUAT, hq);
    rot6(hq, layout, g + 3);

    float v[3], wq[4], w[3];
    for (int i = 0; i < 3; ++i) v[i] = (b[F_POS + i] - a[F_POS + i]) * rate;
    q_rotate_inv(fr + F_QUAT, v, g + 9);
    const float ca[4] =
        {a[F_QUAT], -a[F_QUAT + 1], -a[F_QUAT + 2], -a[F_QUAT + 3]};
    q_mul(b + F_QUAT, ca, wq);
    q_log(wq, w);
    for (int i = 0; i < 3; ++i) w[i] *= rate;
    q_rotate_inv(fr + F_QUAT, w, g + 12);

    for (int j = 0; j < NUM_ACTIONS; ++j) {
      if (j < NUM_OWNED) {
        const int il = D_IL_OF_MJ[j];
        g[15 + j] = fr[F_Q + il];
        g[44 + j] = QD_GOAL_SCALE * (b[F_Q + il] - a[F_Q + il]) * rate;
      } else {
        g[15 + j] = arm_pose[env * POLICY_NUM_MOTOR + j];
        g[44 + j] = 0.0f;
      }
    }
  }
}

__global__ void k_supermimic_vibed_act(
    const float* __restrict__ action,
    const float* __restrict__ arm_pose,
    const int* __restrict__ len,
    float* __restrict__ prev_act,
    int* __restrict__ cursor,
    float* __restrict__ q_target,
    float clip,
    int envs
) {
  const int env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= envs) return;

  const float* a = action + size_t(env) * NUM_ACTIONS;
  float act[NUM_ACTIONS];
  for (int j = 0; j < NUM_ACTIONS; ++j) {
    act[j] = clip > 0.0f ? fminf(fmaxf(a[j], -clip), clip) : a[j];
    prev_act[env * NUM_ACTIONS + j] = act[j];
  }

  for (int j = 0; j < POLICY_NUM_MOTOR; ++j) {
    q_target[env * POLICY_NUM_MOTOR + j] =
        j < NUM_OWNED ? D_DEFAULTS[j] + act[j] * D_SCALE[j]
                      : arm_pose[env * POLICY_NUM_MOTOR + j];
  }

  const int n = len[env];
  if (n > 0) {
    const int c = cursor[env] + 1;
    cursor[env] = c > n - 1 ? n - 1 : c;
  }
}

struct Variant {
  const char* name;
  const char* onnx;
  int layout;
  float clip;
};

const Variant PANHAINAN = {
    "supermimic_panhainan",
    "policies/supermimic_vibed/model_panhainan.onnx",
    ROT6_ROWS,
    0.0f
};
const Variant OMEGAI = {
    "supermimic_omegai",
    "policies/supermimic_vibed/model_omegai.onnx",
    ROT6_COLS,
    20.0f
};

struct Policy : policy_api::Policy {
  const Variant variant;
  std::shared_ptr<policy_api::Engine> planner, actor;

  float *d_ctx = nullptr, *d_mode = nullptr, *d_vel = nullptr;
  float *d_move = nullptr, *d_face = nullptr;
  float *d_qpos = nullptr, *d_nframes = nullptr;
  float *d_motion = nullptr, *d_motion2 = nullptr, *d_gen = nullptr;
  int *d_len = nullptr, *d_cursor = nullptr;
  float *d_psiref = nullptr, *d_yaw0 = nullptr, *d_yaw_off = nullptr;
  float *d_hist = nullptr, *d_goal = nullptr;
  float *d_prev = nullptr, *d_act = nullptr;

  float *d_seed = nullptr, *d_height = nullptr, *d_hst = nullptr;
  float *d_stp = nullptr, *d_sth = nullptr, *d_allow = nullptr;

  long long ticks = 0;
  int ref = mimic_lite::REF_SONIC;
  mimic_lite::Ghost ghost;
  std::string full_name;
  int envs = 0;

  explicit Policy(const Variant& v) : variant(v) {}

  ~Policy() override {
    for (void* p :
         {(void*)d_ctx,     (void*)d_mode, (void*)d_vel,     (void*)d_move,
          (void*)d_face,    (void*)d_qpos, (void*)d_nframes, (void*)d_motion,
          (void*)d_motion2, (void*)d_gen,  (void*)d_len,     (void*)d_cursor,
          (void*)d_psiref,  (void*)d_yaw0, (void*)d_yaw_off, (void*)d_hist,
          (void*)d_goal,    (void*)d_prev, (void*)d_act,     (void*)d_seed,
          (void*)d_height,  (void*)d_hst,  (void*)d_stp,     (void*)d_sth,
          (void*)d_allow}) {
      if (p) cudaFree(p);
    }
  }

  static float* zeros(size_t n) {
    float* p = nullptr;
    cudaMalloc(&p, n * sizeof(float));
    cudaMemset(p, 0, n * sizeof(float));
    return p;
  }

  static float* filled(
      int n,
      float value
  ) {
    float* p = zeros(size_t(n));
    const std::vector<float> h(size_t(n), value);
    cudaMemcpy(p, h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice);
    return p;
  }

  void init(int n) override {
    envs = n;

    if (ref != mimic_lite::REF_SONIC) ghost.init(n, ref);
    if (ref == mimic_lite::REF_SONIC)
      planner = policy_api::engine_make(
          "policies/sonic/model_planner_rows.onnx",
          n,
          {{"context_mujoco_qpos", {-1, CTX_FRAMES, QPOS}},
           {"mode", {-1}},
           {"target_vel", {-1}},
           {"movement_direction", {-1, 3}},
           {"facing_direction", {-1, 3}},
           {"random_seed", {-1}},
           {"height", {-1}},
           {"has_specific_target", {-1, 1}},
           {"specific_target_positions", {-1, CTX_FRAMES, 3}},
           {"specific_target_headings", {-1, CTX_FRAMES}},
           {"allowed_pred_num_tokens", {-1, 11}}},
          {{"mujoco_qpos", {-1, PLAN_FRAMES, QPOS}}, {"num_pred_frames", {-1}}}
      );

    actor = policy_api::engine_make(
        variant.onnx,
        n,
        {{"obs_history", {-1, OBS}}, {"goal", {-1, GOAL}}},
        {{"action", {-1, NUM_ACTIONS}}}
    );

    d_ctx = zeros(size_t(n) * CTX_FRAMES * QPOS);
    d_mode = zeros(size_t(n));
    d_vel = zeros(size_t(n));
    d_move = zeros(size_t(n) * 3);
    d_face = zeros(size_t(n) * 3);
    d_qpos = zeros(size_t(n) * PLAN_FRAMES * QPOS);
    d_nframes = zeros(size_t(n));
    d_motion = zeros(size_t(n) * MAX_FRAMES * FRAME);
    d_motion2 = zeros(size_t(n) * MAX_FRAMES * FRAME);
    d_gen = zeros(size_t(n) * MAX_FRAMES * FRAME);
    cudaMalloc(&d_len, size_t(n) * sizeof(int));
    cudaMemset(d_len, 0, size_t(n) * sizeof(int));
    cudaMalloc(&d_cursor, size_t(n) * sizeof(int));
    cudaMemset(d_cursor, 0, size_t(n) * sizeof(int));
    d_psiref = zeros(size_t(n));
    d_yaw0 = zeros(size_t(n));
    d_yaw_off = zeros(size_t(n));
    d_hist = zeros(size_t(n) * OBS);
    d_goal = zeros(size_t(n) * GOAL);
    d_prev = zeros(size_t(n) * NUM_ACTIONS);
    d_act = zeros(size_t(n) * NUM_ACTIONS);

    d_seed = filled(n, PLAN_SEED);
    d_height = filled(n, -1.0f);
    d_hst = zeros(size_t(n));
    d_stp = zeros(size_t(n) * CTX_FRAMES * 3);
    d_sth = zeros(size_t(n) * CTX_FRAMES);
    d_allow = zeros(size_t(n) * 11);
    std::vector<float> allow(size_t(n) * 11, 0.0f);
    for (int e = 0; e < n; ++e)
      for (int k = 0; k < 6; ++k) allow[size_t(e) * 11 + k] = 1.0f;
    cudaMemcpy(
        d_allow,
        allow.data(),
        allow.size() * sizeof(float),
        cudaMemcpyHostToDevice
    );
  }

  void plan() {
    const float* in[11] = {
        d_ctx,
        d_mode,
        d_vel,
        d_move,
        d_face,
        d_seed,
        d_height,
        d_hst,
        d_stp,
        d_sth,
        d_allow
    };
    float* out[2] = {d_qpos, d_nframes};
    policy_api::engine_run(*planner, in, out, envs);
  }

  void step(const policy_api::Ctx& c) override {
    const int threads = 128, blocks = (envs + threads - 1) / threads;

    if (ref != mimic_lite::REF_SONIC) {
      ghost.advance(c, ticks, d_motion, d_len, d_cursor);
    } else {
      mimic_lite::k_mimic_lite_drive<<<blocks, threads>>>(
          c.cmd,
          c.base_quat,
          d_psiref,
          d_yaw0,
          d_len,
          d_mode,
          d_vel,
          d_move,
          d_face,
          envs
      );

      if (ticks % REPLAN_TICKS == 0) {
        mimic_lite::k_mimic_lite_context<<<blocks, threads>>>(
            d_motion,
            d_len,
            d_cursor,
            c.motor_q,
            d_ctx,
            envs
        );
        plan();
        mimic_lite::k_mimic_lite_apply<<<blocks, threads>>>(
            c.base_quat,
            d_qpos,
            d_nframes,
            d_gen,
            d_motion,
            d_motion2,
            d_len,
            d_cursor,
            d_psiref,
            d_yaw0,
            d_yaw_off,
            envs
        );
        std::swap(d_motion, d_motion2);
      }
    }

    k_supermimic_vibed_obs<<<blocks, threads>>>(
        d_motion,
        d_len,
        d_cursor,
        c.motor_q,
        c.motor_dq,
        c.gyro,
        c.gravity,
        c.base_quat,
        d_yaw_off,
        c.arm_pose,
        d_prev,
        d_hist,
        d_goal,
        variant.layout,
        ticks,
        envs
    );

    const float* in[2] = {d_hist, d_goal};
    float* out[1] = {d_act};
    policy_api::engine_run(*actor, in, out, envs);

    k_supermimic_vibed_act<<<blocks, threads>>>(
        d_act,
        c.arm_pose,
        d_len,
        d_prev,
        d_cursor,
        c.q_target,
        variant.clip,
        envs
    );
    ++ticks;
  }

  const float* kp() const override { return KPS; }
  const float* kd() const override { return KDS; }
  int owned() const override { return NUM_OWNED; }
  policy_api::Limits limits() const override {
    return mimic_lite::reference_limits(ref);
  }
  bool set_reference(int r) override {
    ref = r;
    full_name = std::string(variant.name) + mimic_lite::reference_suffix(r);
    return true;
  }
  const char* name() const override {
    return full_name.empty() ? variant.name : full_name.c_str();
  }
};

std::vector<std::string> names() { return {PANHAINAN.name, OMEGAI.name}; }

std::unique_ptr<policy_api::Policy> make(const std::string& name) {
  for (const Variant* v : {&PANHAINAN, &OMEGAI}) {
    if (name == v->name) return std::make_unique<Policy>(*v);
  }
  return nullptr;
}

}
