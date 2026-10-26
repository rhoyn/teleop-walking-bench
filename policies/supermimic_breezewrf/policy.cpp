namespace supermimic_breezewrf {

using mimic_lite::CTX_FRAMES;
using mimic_lite::D_IL_OF_MJ;
using mimic_lite::F_POS;
using mimic_lite::F_Q;
using mimic_lite::F_QUAT;
using mimic_lite::FRAME;
using mimic_lite::MAX_FRAMES;
using mimic_lite::PLAN_FRAMES;
using mimic_lite::PLAN_SEED;
using mimic_lite::q_mul;
using mimic_lite::q_yaw;
using mimic_lite::QPOS;
using mimic_lite::REPLAN_TICKS;

constexpr int NUM_ACTIONS = 29;
constexpr int NUM_OWNED = 15;
constexpr int PROPRIO = 93;
constexpr int HIST = 10;
constexpr int OBS = PROPRIO * HIST;
constexpr int GOAL_K = 10;
constexpr int NUM_TRACKED = 14;
constexpr int GOAL_FRAME = 2 * NUM_ACTIONS + NUM_TRACKED * 15;
constexpr int GOAL = GOAL_K * GOAL_FRAME;
constexpr int REFERENCE_LEAD = 0;
constexpr int NUM_BODIES = 30;
constexpr float CONTROL_DT = 0.02f;

constexpr int G_JPOS = 0, G_JVEL = 29, G_POS = 58, G_ROT = 100, G_LIN = 184,
              G_ANG = 226;

__device__ const int D_TRACKED[NUM_TRACKED] =
    {0, 2, 4, 6, 8, 10, 12, 15, 17, 19, 22, 24, 26, 29};

#define SUPERMIMIC_BREEZEWRF_DEFAULTS                                      \
  -0.312f, 0.0f, 0.0f, 0.669f, -0.363f, 0.0f, -0.312f, 0.0f, 0.0f, 0.669f, \
      -0.363f, 0.0f, 0.0f, 0.0f, 0.0f, 0.2f, 0.2f, 0.0f, 0.6f, 0.0f, 0.0f, \
      0.0f, 0.2f, -0.2f, 0.0f, 0.6f, 0.0f, 0.0f, 0.0f

__device__ const float D_DEFAULTS[NUM_ACTIONS] = {
    SUPERMIMIC_BREEZEWRF_DEFAULTS
};

constexpr float SCALE_7520_14 = 0.5475464630f;
constexpr float SCALE_7520_22 = 0.3506614664f;
constexpr float SCALE_2X5020 = 0.4385773241519928f;

__device__ const float D_SCALE[NUM_OWNED] = {
    SCALE_7520_14,
    SCALE_7520_22,
    SCALE_7520_14,
    SCALE_7520_22,
    SCALE_2X5020,
    SCALE_2X5020,
    SCALE_7520_14,
    SCALE_7520_22,
    SCALE_7520_14,
    SCALE_7520_22,
    SCALE_2X5020,
    SCALE_2X5020,
    SCALE_7520_14,
    SCALE_2X5020,
    SCALE_2X5020
};

constexpr float KP_7520_14 = 40.17923863450712f;
constexpr float KP_7520_22 = 99.09842777666111f;
constexpr float KP_2X5020 = 28.50124619574858f;
constexpr float KD_7520_14 = 2.557889775413375f;
constexpr float KD_7520_22 = 6.308801853496639f;
constexpr float KD_2X5020 = 1.814445686584846f;

const float KPS[NUM_OWNED] = {
    KP_7520_14,
    KP_7520_22,
    KP_7520_14,
    KP_7520_22,
    KP_2X5020,
    KP_2X5020,
    KP_7520_14,
    KP_7520_22,
    KP_7520_14,
    KP_7520_22,
    KP_2X5020,
    KP_2X5020,
    KP_7520_14,
    KP_2X5020,
    KP_2X5020
};
const float KDS[NUM_OWNED] = {
    KD_7520_14,
    KD_7520_22,
    KD_7520_14,
    KD_7520_22,
    KD_2X5020,
    KD_2X5020,
    KD_7520_14,
    KD_7520_22,
    KD_7520_14,
    KD_7520_22,
    KD_2X5020,
    KD_2X5020,
    KD_7520_14,
    KD_2X5020,
    KD_2X5020
};

__device__ const int D_PARENT[NUM_BODIES] = {-1, 0,  1,  2,  3,  4,  5,  0,
                                             7,  8,  9,  10, 11, 0,  13, 14,
                                             15, 16, 17, 18, 19, 20, 21, 15,
                                             23, 24, 25, 26, 27, 28};
__device__ const float D_BODY_POS[] = {
    0.0f,        0.0f,        0.793f,     0.0f,       0.064452f,    -0.1027f,
    0.0f,        0.052f,      -0.030465f, 0.025001f,  0.0f,         -0.12412f,
    -0.078273f,  0.0021489f,  -0.17734f,  0.0f,       -9.4445e-05f, -0.30001f,
    0.0f,        0.0f,        -0.017558f, 0.0f,       -0.064452f,   -0.1027f,
    0.0f,        -0.052f,     -0.030465f, 0.025001f,  0.0f,         -0.12412f,
    -0.078273f,  -0.0021489f, -0.17734f,  0.0f,       9.4445e-05f,  -0.30001f,
    0.0f,        0.0f,        -0.017558f, 0.0f,       0.0f,         0.0f,
    -0.0039635f, 0.0f,        0.044f,     0.0f,       0.0f,         0.0f,
    0.0039563f,  0.10022f,    0.24778f,   0.0f,       0.038f,       -0.013831f,
    0.0f,        0.00624f,    -0.1032f,   0.015783f,  0.0f,         -0.080518f,
    0.1f,        0.00188791f, -0.01f,     0.038f,     0.0f,         0.0f,
    0.046f,      0.0f,        0.0f,       0.0039563f, -0.10021f,    0.24778f,
    0.0f,        -0.038f,     -0.013831f, 0.0f,       -0.00624f,    -0.1032f,
    0.015783f,   0.0f,        -0.080518f, 0.1f,       -0.00188791f, -0.01f,
    0.038f,      0.0f,        0.0f,       0.046f,     0.0f,         0.0f
};
__device__ const float D_BODY_QUAT[] = {
    1.0f,         0.0f,          0.0f,           0.0f,
    1.0f,         0.0f,          0.0f,           0.0f,
    0.996178686f, 0.0f,          -0.0873385724f, 0.0f,
    1.0f,         0.0f,          0.0f,           0.0f,
    0.996178686f, 0.0f,          0.0873385724f,  0.0f,
    1.0f,         0.0f,          0.0f,           0.0f,
    1.0f,         0.0f,          0.0f,           0.0f,
    1.0f,         0.0f,          0.0f,           0.0f,
    0.996178686f, 0.0f,          -0.0873385724f, 0.0f,
    1.0f,         0.0f,          0.0f,           0.0f,
    0.996178686f, 0.0f,          0.0873385724f,  0.0f,
    1.0f,         0.0f,          0.0f,           0.0f,
    1.0f,         0.0f,          0.0f,           0.0f,
    1.0f,         0.0f,          0.0f,           0.0f,
    1.0f,         0.0f,          0.0f,           0.0f,
    1.0f,         0.0f,          0.0f,           0.0f,
    0.99026414f,  0.13920102f,   1.3872202e-05f, -9.86868139e-05f,
    0.990268219f, -0.139172031f, 0.0f,           0.0f,
    1.0f,         0.0f,          0.0f,           0.0f,
    1.0f,         0.0f,          0.0f,           0.0f,
    1.0f,         0.0f,          0.0f,           0.0f,
    1.0f,         0.0f,          0.0f,           0.0f,
    1.0f,         0.0f,          0.0f,           0.0f,
    0.99026414f,  -0.13920102f,  1.3872202e-05f, 9.86868139e-05f,
    0.990268219f, 0.139172031f,  0.0f,           0.0f,
    1.0f,         0.0f,          0.0f,           0.0f,
    1.0f,         0.0f,          0.0f,           0.0f,
    1.0f,         0.0f,          0.0f,           0.0f,
    1.0f,         0.0f,          0.0f,           0.0f,
    1.0f,         0.0f,          0.0f,           0.0f
};
__device__ const float D_AXIS[] = {
    0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f,
    0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f,
    1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f,
    0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f,
    1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f,
    0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f,
    0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f
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

__device__ inline void cross3(
    const float* a,
    const float* b,
    float* out
) {
  out[0] = a[1] * b[2] - a[2] * b[1];
  out[1] = a[2] * b[0] - a[0] * b[2];
  out[2] = a[0] * b[1] - a[1] * b[0];
}

__device__ inline int clamp_frame(
    int f,
    int len
) {
  const int last = len > 0 ? len - 1 : 0;
  return f < 0 ? 0 : (f > last ? last : f);
}

__device__ inline void ang_vel(
    const float* a,
    const float* b,
    float dt,
    float* out
) {
  const float ia[4] = {a[0], -a[1], -a[2], -a[3]};
  float r[4];
  q_mul(b, ia, r);
  if (r[0] < 0.0f)
    for (int k = 0; k < 4; ++k) r[k] = -r[k];
  const float s = sqrtf(r[1] * r[1] + r[2] * r[2] + r[3] * r[3]);
  const float angle = 2.0f * atan2f(s, r[0]);
  const float g = s > 1e-7f ? angle / s : 2.0f;
  for (int k = 0; k < 3; ++k) out[k] = g * r[1 + k] / dt;
}

__device__ void frame_features(
    const float* m,
    int n,
    int f,
    const float* arm,
    const float* origin,
    const float* inv_yaw,
    float* g
) {
  const int ip = clamp_frame(f - 1, n), in = clamp_frame(f + 1, n);
  const float* fr = m + size_t(clamp_frame(f, n)) * FRAME;
  const float* fp = m + size_t(ip) * FRAME;
  const float* fn = m + size_t(in) * FRAME;
  const float span = float(in - ip) * CONTROL_DT;
  const float inv_span = span > 0.0f ? 1.0f / span : 0.0f;

  float q[NUM_ACTIONS], qd[NUM_ACTIONS];
  for (int j = 0; j < NUM_ACTIONS; ++j) {
    if (j < NUM_OWNED) {
      const int il = F_Q + D_IL_OF_MJ[j];
      q[j] = fr[il];
      qd[j] = (fn[il] - fp[il]) * inv_span;
    } else {
      q[j] = arm[j];
      qd[j] = 0.0f;
    }
    g[G_JPOS + j] = (q[j] - D_DEFAULTS[j]) / 0.3f;
    g[G_JVEL + j] = qd[j] / 5.0f;
  }

  float pos[NUM_BODIES][3], quat[NUM_BODIES][4];
  float lin[NUM_BODIES][3], ang[NUM_BODIES][3];
  for (int k = 0; k < 3; ++k) {
    pos[0][k] = fr[F_POS + k];
    lin[0][k] = (fn[F_POS + k] - fp[F_POS + k]) * inv_span;
  }
  for (int k = 0; k < 4; ++k) quat[0][k] = fr[F_QUAT + k];
  if (span > 0.0f) {
    ang_vel(fp + F_QUAT, fn + F_QUAT, span, ang[0]);
  } else {
    ang[0][0] = ang[0][1] = ang[0][2] = 0.0f;
  }

  for (int b = 1; b < NUM_BODIES; ++b) {
    const int p = D_PARENT[b], j = b - 1;
    float off[3], base[4], axis[3], rot[4], w[3];
    q_rotate(quat[p], D_BODY_POS + 3 * b, off);
    q_mul(quat[p], D_BODY_QUAT + 4 * b, base);
    q_rotate(base, D_AXIS + 3 * j, axis);
    const float c = cosf(0.5f * q[j]), s = sinf(0.5f * q[j]);
    rot[0] = c;
    for (int k = 0; k < 3; ++k) rot[1 + k] = s * D_AXIS[3 * j + k];
    q_mul(base, rot, quat[b]);
    cross3(ang[p], off, w);
    for (int k = 0; k < 3; ++k) {
      pos[b][k] = pos[p][k] + off[k];
      lin[b][k] = lin[p][k] + w[k];
      ang[b][k] = ang[p][k] + axis[k] * qd[j];
    }
  }

  const float ex[3] = {1.0f, 0.0f, 0.0f}, ey[3] = {0.0f, 1.0f, 0.0f};
  for (int t = 0; t < NUM_TRACKED; ++t) {
    const int b = D_TRACKED[t];
    float d[3], v[3], r[4];
    for (int k = 0; k < 3; ++k) d[k] = pos[b][k] - origin[k];
    q_rotate(inv_yaw, d, v);
    for (int k = 0; k < 3; ++k) g[G_POS + 3 * t + k] = v[k] / 0.3f;
    q_mul(inv_yaw, quat[b], r);
    q_rotate(r, ex, g + G_ROT + 6 * t);
    q_rotate(r, ey, g + G_ROT + 6 * t + 3);
    q_rotate(inv_yaw, lin[b], g + G_LIN + 3 * t);
    q_rotate(inv_yaw, ang[b], v);
    for (int k = 0; k < 3; ++k) g[G_ANG + 3 * t + k] = v[k] / 3.14f;
  }
}

__global__ void k_supermimic_breezewrf_obs(
    const float* __restrict__ motion,
    const int* __restrict__ len,
    const int* __restrict__ cursor,
    const float* __restrict__ motor_q,
    const float* __restrict__ motor_dq,
    const float* __restrict__ gyro,
    const float* __restrict__ gravity,
    const float* __restrict__ arm_pose,
    const float* __restrict__ prev_act,
    float* __restrict__ hist,
    float* __restrict__ goal,
    long long tick,
    int envs
) {
  const int env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= envs) return;

  const float* mq = motor_q + env * POLICY_NUM_MOTOR;
  const float* mdq = motor_dq + env * POLICY_NUM_MOTOR;
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
  if (tick == 0) {
    for (int i = 0; i < HIST - 1; ++i) {
      for (int k = 0; k < PROPRIO; ++k) h[i * PROPRIO + k] = p[k];
    }
  }

  const float* m = motion + size_t(env) * MAX_FRAMES * FRAME;
  const int n = len[env];
  const int first = cursor[env] + REFERENCE_LEAD;
  const float* f0 = m + size_t(clamp_frame(first, n)) * FRAME;
  const float origin[3] = {f0[F_POS + 0], f0[F_POS + 1], 0.0f};
  const float yaw = q_yaw(f0 + F_QUAT);
  const float inv_yaw[4] = {cosf(0.5f * yaw), 0.0f, 0.0f, -sinf(0.5f * yaw)};
  const float* arm = arm_pose + env * POLICY_NUM_MOTOR;
  float* o = goal + size_t(env) * GOAL;
  for (int k = 0; k < GOAL_K; ++k) {
    frame_features(m, n, first + k, arm, origin, inv_yaw, o + k * GOAL_FRAME);
  }
}

__global__ void k_supermimic_breezewrf_act(
    const float* __restrict__ action,
    const float* __restrict__ arm_pose,
    const int* __restrict__ len,
    float* __restrict__ prev_act,
    int* __restrict__ cursor,
    float* __restrict__ q_target,
    int envs
) {
  const int env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= envs) return;

  const float* a = action + size_t(env) * NUM_ACTIONS;
  for (int j = 0; j < NUM_ACTIONS; ++j) prev_act[env * NUM_ACTIONS + j] = a[j];

  for (int j = 0; j < POLICY_NUM_MOTOR; ++j) {
    q_target[env * POLICY_NUM_MOTOR + j] =
        j < NUM_OWNED ? D_DEFAULTS[j] + a[j] * D_SCALE[j]
                      : arm_pose[env * POLICY_NUM_MOTOR + j];
  }

  const int n = len[env];
  if (n > 0) {
    const int c = cursor[env] + 1;
    cursor[env] = c > n - 1 ? n - 1 : c;
  }
}

struct Policy : policy_api::Policy {
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
        "policies/supermimic_breezewrf/model.onnx",
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
      supermimic::k_supermimic_face<<<blocks, threads>>>(
          c.cmd,
          d_motion,
          d_len,
          d_cursor,
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

    k_supermimic_breezewrf_obs<<<blocks, threads>>>(
        d_motion,
        d_len,
        d_cursor,
        c.motor_q,
        c.motor_dq,
        c.gyro,
        c.gravity,
        c.arm_pose,
        d_prev,
        d_hist,
        d_goal,
        ticks,
        envs
    );

    const float* in[2] = {d_hist, d_goal};
    float* out[1] = {d_act};
    policy_api::engine_run(*actor, in, out, envs);

    k_supermimic_breezewrf_act<<<blocks, threads>>>(
        d_act,
        c.arm_pose,
        d_len,
        d_prev,
        d_cursor,
        c.q_target,
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
    full_name =
        std::string("supermimic_breezewrf") + mimic_lite::reference_suffix(r);
    return true;
  }
  const char* name() const override {
    return full_name.empty() ? "supermimic_breezewrf" : full_name.c_str();
  }
};

}
