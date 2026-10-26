namespace supermimic {

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
constexpr int GOAL_FRAME = 68;
constexpr int ANCHOR = 9;
constexpr float CONTROL_DT = 0.02f;
constexpr float LEASH_M = 0.25f;
constexpr int WAIST_YAW = 12, WAIST_ROLL = 13, WAIST_PITCH = 14;

constexpr int TERMS = 5;
__device__ const int D_TERM_LO[TERMS + 1] = {0, 29, 58, 61, 64, 93};

__device__ const float D_TORSO_OFF[3] = {-0.0039635f, 0.0f, 0.044f};

struct Variant {
  const char* name;
  const char* onnx;
  int lead;
  bool anchor;
  bool steer;
};

const Variant VARIANTS[] = {
    {"supermimic_kangzejian",
     "policies/supermimic/model_kangzejian.onnx",
     1,
     false,
     true},
    {"supermimic_kyrieyoung",
     "policies/supermimic/model_kyrieyoung.onnx",
     0,
     false,
     true},
    {"supermimic_rayzhao",
     "policies/supermimic/model_rayzhao.onnx",
     1,
     false,
     true},
    {"supermimic_dyn", "policies/supermimic/model_dyn.onnx", 1, true, false},
    {"supermimic_m5", "policies/supermimic/model_m5.onnx", 1, true, false},
};

__device__ const float D_DEFAULTS[NUM_ACTIONS] = {
    -0.312f, 0.0f, 0.0f, 0.669f, -0.363f, 0.0f, -0.312f, 0.0f, 0.0f, 0.669f,
    -0.363f, 0.0f, 0.0f, 0.0f,   0.0f,    0.2f, 0.2f,    0.0f, 0.6f, 0.0f,
    0.0f,    0.0f, 0.2f, -0.2f,  0.0f,    0.6f, 0.0f,    0.0f, 0.0f
};

constexpr float S_7520_14 = 0.5475464463233948f;
constexpr float S_7520_22 = 0.3506614565849304f;
constexpr float S_5020X2 = 0.4385773241519928f;

__device__ const float D_SCALE[NUM_OWNED] = {
    S_7520_14,
    S_7520_22,
    S_7520_14,
    S_7520_22,
    S_5020X2,
    S_5020X2,
    S_7520_14,
    S_7520_22,
    S_7520_14,
    S_7520_22,
    S_5020X2,
    S_5020X2,
    S_7520_14,
    S_5020X2,
    S_5020X2
};

constexpr float KP_7520_14 = 40.17923863450712f;
constexpr float KP_7520_22 = 99.09842777666111f;
constexpr float KP_5020X2 = 28.50124619574858f;
constexpr float KD_7520_14 = 2.557889775413375f;
constexpr float KD_7520_22 = 6.308801853496639f;
constexpr float KD_5020X2 = 1.814445686584846f;

const float KPS[NUM_OWNED] = {
    KP_7520_14,
    KP_7520_22,
    KP_7520_14,
    KP_7520_22,
    KP_5020X2,
    KP_5020X2,
    KP_7520_14,
    KP_7520_22,
    KP_7520_14,
    KP_7520_22,
    KP_5020X2,
    KP_5020X2,
    KP_7520_14,
    KP_5020X2,
    KP_5020X2
};
const float KDS[NUM_OWNED] = {
    KD_7520_14,
    KD_7520_22,
    KD_7520_14,
    KD_7520_22,
    KD_5020X2,
    KD_5020X2,
    KD_7520_14,
    KD_7520_22,
    KD_7520_14,
    KD_7520_22,
    KD_5020X2,
    KD_5020X2,
    KD_7520_14,
    KD_5020X2,
    KD_5020X2
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

__device__ inline void q_axis(
    int axis,
    float angle,
    float* out
) {
  out[0] = cosf(0.5f * angle);
  out[1] = out[2] = out[3] = 0.0f;
  out[1 + axis] = sinf(0.5f * angle);
}

__device__ inline void rotvec(
    const float* q,
    float* out
) {
  const float sgn = q[0] < 0.0f ? -1.0f : 1.0f;
  const float w = sgn * q[0];
  const float x = sgn * q[1], y = sgn * q[2], z = sgn * q[3];
  const float s = sqrtf(x * x + y * y + z * z);
  const float k = s > 1e-9f ? 2.0f * atan2f(s, w) / s : 2.0f;
  out[0] = x * k;
  out[1] = y * k;
  out[2] = z * k;
}

__device__ inline void rot6_cols(
    const float* q,
    float* out
) {
  const float w = q[0], x = q[1], y = q[2], z = q[3];
  out[0] = 1.0f - 2.0f * (y * y + z * z);
  out[1] = 2.0f * (x * y - w * z);
  out[2] = 2.0f * (x * y + w * z);
  out[3] = 1.0f - 2.0f * (x * x + z * z);
  out[4] = 2.0f * (x * z - w * y);
  out[5] = 2.0f * (y * z + w * x);
}

__device__ inline void torso_of(
    const float* pelvis_pos,
    const float* pelvis_quat,
    float yaw,
    float roll,
    float pitch,
    float* pos,
    float* quat
) {
  float qy[4], qr[4], qp[4], a[4], b[4];
  q_axis(2, yaw, qy);
  q_axis(0, roll, qr);
  q_axis(1, pitch, qp);
  q_mul(pelvis_quat, qy, a);
  q_mul(a, qr, b);
  q_mul(b, qp, quat);
  float in_pelvis[3], world[3];
  q_rotate(qy, D_TORSO_OFF, in_pelvis);
  q_rotate(pelvis_quat, in_pelvis, world);
  for (int k = 0; k < 3; ++k) pos[k] = pelvis_pos[k] + world[k];
}

__device__ inline int clamp_frame(
    int f,
    int len
) {
  const int last = len > 0 ? len - 1 : 0;
  return f < 0 ? 0 : (f > last ? last : f);
}

__global__ void k_supermimic_tie(
    const float* __restrict__ base_pos,
    const float* __restrict__ motion,
    const int* __restrict__ len,
    float* __restrict__ tie,
    int envs
) {
  const int env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= envs) return;
  float* t = tie + env * 7;
  if (t[6] != 0.0f || len[env] == 0) return;
  const float* f0 = motion + size_t(env) * MAX_FRAMES * FRAME;
  t[0] = base_pos[env * 3 + 0];
  t[1] = base_pos[env * 3 + 1];
  t[2] = f0[F_POS + 0];
  t[3] = f0[F_POS + 1];
  t[4] = 0.0f;
  t[5] = 0.0f;
  t[6] = 1.0f;
}

__global__ void k_supermimic_obs(
    const float* __restrict__ motion,
    const int* __restrict__ len,
    const int* __restrict__ cursor,
    const float* __restrict__ motor_q,
    const float* __restrict__ motor_dq,
    const float* __restrict__ gyro,
    const float* __restrict__ gravity,
    const float* __restrict__ base_quat,
    const float* __restrict__ base_pos,
    const float* __restrict__ yaw_off,
    const float* __restrict__ arm_pose,
    const float* __restrict__ prev_act,
    float* __restrict__ tie,
    float* __restrict__ hist,
    float* __restrict__ obs,
    float* __restrict__ goal,
    int lead,
    bool anchor,
    int goal_dim,
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

  float* o = obs + size_t(env) * OBS;
  for (int t = 0; t < TERMS; ++t) {
    const int lo = D_TERM_LO[t], d = D_TERM_LO[t + 1] - lo;
    for (int i = 0; i < HIST; ++i) {
      for (int k = 0; k < d; ++k) {
        o[lo * HIST + i * d + k] = h[i * PROPRIO + lo + k];
      }
    }
  }

  const float* m = motion + size_t(env) * MAX_FRAMES * FRAME;
  const int n = len[env];
  const int first = cursor[env] + lead;
  float* gl = goal + size_t(env) * goal_dim;
  for (int k = 0; k < GOAL_K; ++k) {
    const int fi = clamp_frame(first + k, n);
    const int pi = clamp_frame(fi - 1, n), ni = clamp_frame(fi + 1, n);
    const float span = float(ni - pi) * CONTROL_DT;
    const float inv = span > 0.0f ? 1.0f / span : 0.0f;
    const float* fr = m + size_t(fi) * FRAME;
    const float* prev = m + size_t(pi) * FRAME;
    const float* next = m + size_t(ni) * FRAME;
    float* g = gl + k * GOAL_FRAME;

    for (int j = 0; j < NUM_ACTIONS; ++j) {
      if (j < NUM_OWNED) {
        const int il = F_Q + D_IL_OF_MJ[j];
        g[j] = fr[il] - D_DEFAULTS[j];
        g[29 + j] = (next[il] - prev[il]) * inv;
      } else {
        g[j] = arm_pose[env * POLICY_NUM_MOTOR + j] - D_DEFAULTS[j];
        g[29 + j] = 0.0f;
      }
    }

    const float down[3] = {0.0f, 0.0f, -1.0f};
    q_rotate_inv(fr + F_QUAT, down, g + 58);

    const float conj[4] =
        {prev[F_QUAT], -prev[F_QUAT + 1], -prev[F_QUAT + 2], -prev[F_QUAT + 3]};
    float rel[4], w_world[3];
    q_mul(next + F_QUAT, conj, rel);
    rotvec(rel, w_world);
    for (int c = 0; c < 3; ++c) w_world[c] *= inv;
    q_rotate_inv(fr + F_QUAT, w_world, g + 61);

    const float vx = (next[F_POS + 0] - prev[F_POS + 0]) * inv;
    const float vy = (next[F_POS + 1] - prev[F_POS + 1]) * inv;
    const float vz = (next[F_POS + 2] - prev[F_POS + 2]) * inv;
    const float yaw = q_yaw(fr + F_QUAT);
    const float c = cosf(yaw), s = sinf(yaw);
    g[64] = c * vx + s * vy;
    g[65] = -s * vx + c * vy;
    g[66] = vz;
    g[67] = fr[F_POS + 2];
  }
  if (!anchor) return;

  const float* f0 = m + size_t(clamp_frame(first, n)) * FRAME;
  float ref_pos[3], ref_quat[4];
  torso_of(
      f0 + F_POS,
      f0 + F_QUAT,
      f0[F_Q + D_IL_OF_MJ[WAIST_YAW]],
      f0[F_Q + D_IL_OF_MJ[WAIST_ROLL]],
      f0[F_Q + D_IL_OF_MJ[WAIST_PITCH]],
      ref_pos,
      ref_quat
  );

  float* t = tie + env * 7;
  const float yo = yaw_off[env];
  const float cy = cosf(yo), sy = sinf(yo);
  const float off[4] = {cosf(0.5f * yo), 0.0f, 0.0f, sinf(0.5f * yo)};
  float pel_quat[4];
  q_mul(off, base_quat + env * 4, pel_quat);
  const float* bp = base_pos + env * 3;
  const float dx = bp[0] - t[0], dy = bp[1] - t[1];
  float pel_pos[3] = {
      cy * dx - sy * dy + t[2] + t[4],
      sy * dx + cy * dy + t[3] + t[5],
      bp[2]
  };
  float rob_pos[3], rob_quat[4];
  torso_of(
      pel_pos,
      pel_quat,
      mq[WAIST_YAW],
      mq[WAIST_ROLL],
      mq[WAIST_PITCH],
      rob_pos,
      rob_quat
  );

  float ex = ref_pos[0] - rob_pos[0], ey = ref_pos[1] - rob_pos[1];
  const float e = sqrtf(ex * ex + ey * ey);
  if (e > LEASH_M) {
    const float slip = 1.0f - LEASH_M / e;
    t[4] += ex * slip;
    t[5] += ey * slip;
    rob_pos[0] += ex * slip;
    rob_pos[1] += ey * slip;
  }

  const float diff[3] = {
      ref_pos[0] - rob_pos[0],
      ref_pos[1] - rob_pos[1],
      ref_pos[2] - rob_pos[2]
  };
  float* a = gl + GOAL_K * GOAL_FRAME;
  q_rotate_inv(rob_quat, diff, a);
  const float inv_rob[4] =
      {rob_quat[0], -rob_quat[1], -rob_quat[2], -rob_quat[3]};
  float rel[4];
  q_mul(inv_rob, ref_quat, rel);
  rot6_cols(rel, a + 3);
}

__global__ void k_supermimic_face(
    const float* __restrict__ cmd,
    const float* __restrict__ motion,
    const int* __restrict__ len,
    const int* __restrict__ cursor,
    float* __restrict__ move,
    float* __restrict__ face,
    int envs
) {
  const int env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= envs || len[env] == 0) return;

  const float* m = motion + size_t(env) * MAX_FRAMES * FRAME;
  const float* now = m + size_t(clamp_frame(cursor[env], len[env])) * FRAME;
  const float* d = cmd + env * 3;
  const float ref_yaw = q_yaw(now + F_QUAT);
  const float yaw = ref_yaw + d[2] * mimic_lite::FACE_LEAD_S;
  face[env * 3 + 0] = cosf(yaw);
  face[env * 3 + 1] = sinf(yaw);
  if (move[env * 3 + 0] != 0.0f || move[env * 3 + 1] != 0.0f) {
    const float course = ref_yaw + atan2f(d[1], d[0]);
    move[env * 3 + 0] = cosf(course);
    move[env * 3 + 1] = sinf(course);
  }
}

__global__ void k_supermimic_act(
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
  Variant v;
  int goal_dim;
  std::shared_ptr<policy_api::Engine> planner, actor;

  float *d_ctx = nullptr, *d_mode = nullptr, *d_vel = nullptr;
  float *d_move = nullptr, *d_face = nullptr;
  float *d_qpos = nullptr, *d_nframes = nullptr;
  float *d_motion = nullptr, *d_motion2 = nullptr, *d_gen = nullptr;
  int *d_len = nullptr, *d_cursor = nullptr;
  float *d_psiref = nullptr, *d_yaw0 = nullptr, *d_yaw_off = nullptr;
  float *d_tie = nullptr, *d_hist = nullptr, *d_obs = nullptr;
  float *d_goal = nullptr, *d_prev = nullptr, *d_act = nullptr;

  float *d_seed = nullptr, *d_height = nullptr, *d_hst = nullptr;
  float *d_stp = nullptr, *d_sth = nullptr, *d_allow = nullptr;

  long long ticks = 0;
  int ref = mimic_lite::REF_SONIC;
  mimic_lite::Ghost ghost;
  std::string full_name;
  int envs = 0;

  explicit Policy(const Variant& variant)
      : v(variant),
        goal_dim(GOAL_K * GOAL_FRAME + (variant.anchor ? ANCHOR : 0)) {}

  ~Policy() override {
    for (void* p :
         {(void*)d_ctx,     (void*)d_mode, (void*)d_vel,     (void*)d_move,
          (void*)d_face,    (void*)d_qpos, (void*)d_nframes, (void*)d_motion,
          (void*)d_motion2, (void*)d_gen,  (void*)d_len,     (void*)d_cursor,
          (void*)d_psiref,  (void*)d_yaw0, (void*)d_yaw_off, (void*)d_tie,
          (void*)d_hist,    (void*)d_obs,  (void*)d_goal,    (void*)d_prev,
          (void*)d_act,     (void*)d_seed, (void*)d_height,  (void*)d_hst,
          (void*)d_stp,     (void*)d_sth,  (void*)d_allow}) {
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
        v.onnx,
        n,
        {{"obs_history", {-1, OBS}}, {"goal", {-1, goal_dim}}},
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
    d_tie = zeros(size_t(n) * 7);
    d_hist = zeros(size_t(n) * OBS);
    d_obs = zeros(size_t(n) * OBS);
    d_goal = zeros(size_t(n) * goal_dim);
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
      if (v.anchor)
        k_supermimic_tie<<<blocks, threads>>>(
            c.base_pos,
            d_motion,
            d_len,
            d_tie,
            envs
        );
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
      if (v.steer) {
        k_supermimic_face<<<blocks, threads>>>(
            c.cmd,
            d_motion,
            d_len,
            d_cursor,
            d_move,
            d_face,
            envs
        );
      }

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
        if (v.anchor) {
          k_supermimic_tie<<<blocks, threads>>>(
              c.base_pos,
              d_motion,
              d_len,
              d_tie,
              envs
          );
        }
      }
    }

    k_supermimic_obs<<<blocks, threads>>>(
        d_motion,
        d_len,
        d_cursor,
        c.motor_q,
        c.motor_dq,
        c.gyro,
        c.gravity,
        c.base_quat,
        c.base_pos,
        d_yaw_off,
        c.arm_pose,
        d_prev,
        d_tie,
        d_hist,
        d_obs,
        d_goal,
        v.lead,
        v.anchor,
        goal_dim,
        ticks,
        envs
    );

    const float* in[2] = {d_obs, d_goal};
    float* out[1] = {d_act};
    policy_api::engine_run(*actor, in, out, envs);

    k_supermimic_act<<<blocks, threads>>>(
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
    full_name = std::string(v.name) + mimic_lite::reference_suffix(r);
    return true;
  }
  const char* name() const override {
    return full_name.empty() ? v.name : full_name.c_str();
  }
};

std::vector<std::string> names() {
  std::vector<std::string> all;
  for (const Variant& v : VARIANTS) all.emplace_back(v.name);
  return all;
}

std::unique_ptr<policy_api::Policy> make(const std::string& name) {
  for (const Variant& v : VARIANTS) {
    if (name == v.name) return std::make_unique<Policy>(v);
  }
  return nullptr;
}

}
