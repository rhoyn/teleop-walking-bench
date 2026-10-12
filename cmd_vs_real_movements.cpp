#define STB_TRUETYPE_IMPLEMENTATION
#include <stb_truetype.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <optional>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace canvas {

constexpr int TILE = 800;
constexpr int PX_PER_SCALE = 9;
constexpr int TITLE_SCALE = 4;
constexpr int LABEL_SCALE = 2;
constexpr int HEADER_H = 80;
constexpr int STRIP_H = 20;

struct RGB {
  unsigned char r, g, b;
};

constexpr RGB BG = {46, 111, 64};
constexpr RGB WHITE = {255, 255, 255};
constexpr RGB BLACK = {0, 0, 0};

struct Image {
  int w = TILE, h = TILE;
  std::vector<unsigned char> px;
  int clip_x0 = 0, clip_y0 = 0, clip_x1 = TILE, clip_y1 = TILE;
};

struct Clip {
  Image& im;
  int x0, y0, x1, y1;
  Clip(
      Image& i,
      int cx0,
      int cy0,
      int cx1,
      int cy1
  )
      : im(i),
        x0(i.clip_x0),
        y0(i.clip_y0),
        x1(i.clip_x1),
        y1(i.clip_y1) {
    im.clip_x0 = cx0;
    im.clip_y0 = cy0;
    im.clip_x1 = cx1;
    im.clip_y1 = cy1;
  }
  ~Clip() {
    im.clip_x0 = x0;
    im.clip_y0 = y0;
    im.clip_x1 = x1;
    im.clip_y1 = y1;
  }
};

struct Axes {
  int px = 0, py = 0, pw = 1, ph = 1;
  double x0 = 0, x1 = 1, y0 = 0, y1 = 1;
};

struct Font {
  std::vector<unsigned char> data;
  stbtt_fontinfo info;
  bool ok = false;
};

Font& font() {
  static Font f;
  static bool tried = false;
  if (tried) return f;
  tried = true;
  std::ifstream in("assets/JetBrainsMono.ttf", std::ios::binary);
  f.data.assign(
      std::istreambuf_iterator<char>(in),
      std::istreambuf_iterator<char>()
  );
  f.ok = !f.data.empty() && stbtt_InitFont(
                                &f.info,
                                f.data.data(),
                                stbtt_GetFontOffsetForIndex(f.data.data(), 0)
                            );
  if (!f.ok) {
    std::fprintf(
        stderr,
        "cmd_vs_real_movements: cannot read assets/JetBrainsMono.ttf, "
        "charts will have no text\n"
    );
  }
  return f;
}

bool clipped(
    const Image& im,
    int x,
    int y
) {
  return x < 0 || y < 0 || x >= im.w || y >= im.h || x < im.clip_x0 ||
         y < im.clip_y0 || x >= im.clip_x1 || y >= im.clip_y1;
}

void put(
    Image& im,
    int x,
    int y,
    RGB c
) {
  if (clipped(im, x, y)) return;
  const size_t i = (size_t(y) * im.w + x) * 3;
  im.px[i] = c.r;
  im.px[i + 1] = c.g;
  im.px[i + 2] = c.b;
}

void blend(
    Image& im,
    int x,
    int y,
    unsigned char a,
    RGB c
) {
  if (a < 16 || clipped(im, x, y)) return;
  const size_t o = (size_t(y) * im.w + x) * 3;
  const float t = float(a) / 255.0f;
  im.px[o] = (unsigned char)(im.px[o] * (1 - t) + c.r * t);
  im.px[o + 1] = (unsigned char)(im.px[o + 1] * (1 - t) + c.g * t);
  im.px[o + 2] = (unsigned char)(im.px[o + 2] * (1 - t) + c.b * t);
}

void fill_rect(
    Image& im,
    int x,
    int y,
    int w,
    int h,
    RGB c
) {
  for (int j = y; j < y + h; j++)
    for (int i = x; i < x + w; i++) put(im, i, j, c);
}

void rect(
    Image& im,
    int x,
    int y,
    int w,
    int h,
    RGB c
) {
  for (int i = x; i < x + w; i++) {
    put(im, i, y, c);
    put(im, i, y + h - 1, c);
  }
  for (int j = y; j < y + h; j++) {
    put(im, x, j, c);
    put(im, x + w - 1, j, c);
  }
}

void line(
    Image& im,
    int x0,
    int y0,
    int x1,
    int y1,
    RGB c
) {
  int dx = std::abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
  int dy = -std::abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
  int err = dx + dy;
  for (;;) {
    put(im, x0, y0, c);
    if (x0 == x1 && y0 == y1) break;
    const int e2 = 2 * err;
    if (e2 >= dy) {
      err += dy;
      x0 += sx;
    }
    if (e2 <= dx) {
      err += dx;
      y0 += sy;
    }
  }
}

int text_height(int scale) { return scale * PX_PER_SCALE; }

int text_width(
    const std::string& str,
    int scale
) {
  Font& f = font();
  if (!f.ok) return int(str.size()) * scale * PX_PER_SCALE / 2;
  const float sf =
      stbtt_ScaleForPixelHeight(&f.info, float(text_height(scale)));
  float w = 0;
  for (char ch : str) {
    int aw = 0, lsb = 0;
    stbtt_GetCodepointHMetrics(&f.info, (unsigned char)ch, &aw, &lsb);
    w += float(aw) * sf;
  }
  return int(w);
}

std::vector<unsigned char> text_mask(
    const std::string& str,
    int scale,
    int& w,
    int& h
) {
  std::vector<unsigned char> mask;
  Font& f = font();
  w = h = 0;
  if (!f.ok || str.empty()) return mask;
  const float sf =
      stbtt_ScaleForPixelHeight(&f.info, float(text_height(scale)));
  int ascent = 0, descent = 0, gap = 0;
  stbtt_GetFontVMetrics(&f.info, &ascent, &descent, &gap);
  const int baseline = int(float(ascent) * sf);
  h = int(float(ascent - descent) * sf) + 2;
  float total = 0;
  for (char ch : str) {
    int aw = 0, lsb = 0;
    stbtt_GetCodepointHMetrics(&f.info, (unsigned char)ch, &aw, &lsb);
    total += float(aw) * sf;
  }
  w = int(total) + 2;
  if (w <= 0 || h <= 0) return mask;
  mask.assign(size_t(w) * h, 0);
  float xpos = 0;
  for (char ch : str) {
    const int c = (unsigned char)ch;
    int aw = 0, lsb = 0;
    stbtt_GetCodepointHMetrics(&f.info, c, &aw, &lsb);
    int gw = 0, gh = 0, gx = 0, gy = 0;
    unsigned char* bmp =
        stbtt_GetCodepointBitmap(&f.info, sf, sf, c, &gw, &gh, &gx, &gy);
    if (bmp) {
      for (int j = 0; j < gh; j++)
        for (int i = 0; i < gw; i++) {
          const int dx = int(xpos) + gx + i;
          const int dy = baseline + gy + j;
          if (dx < 0 || dy < 0 || dx >= w || dy >= h) continue;
          unsigned char& d = mask[size_t(dy) * w + dx];
          d = std::max(d, bmp[j * gw + i]);
        }
      stbtt_FreeBitmap(bmp, nullptr);
    }
    xpos += float(aw) * sf;
  }
  return mask;
}

void text(
    Image& im,
    int x,
    int y,
    const std::string& str,
    RGB c,
    int scale
) {
  int w = 0, h = 0;
  const std::vector<unsigned char> m = text_mask(str, scale, w, h);
  for (int j = 0; j < h; j++)
    for (int i = 0; i < w; i++)
      blend(im, x + i, y + j, m[size_t(j) * w + i], c);
}

void text_rot(
    Image& im,
    int x,
    int y,
    const std::string& str,
    RGB c,
    int scale
) {
  int w = 0, h = 0;
  const std::vector<unsigned char> m = text_mask(str, scale, w, h);
  for (int j = 0; j < h; j++)
    for (int i = 0; i < w; i++)
      blend(im, x + j, y - i, m[size_t(j) * w + i], c);
}

int sx(
    const Axes& ax,
    double x
) {
  return ax.px + int((x - ax.x0) / (ax.x1 - ax.x0) * (ax.pw - 1));
}

int sy(
    const Axes& ax,
    double y
) {
  return ax.py + ax.ph - 1 - int((y - ax.y0) / (ax.y1 - ax.y0) * (ax.ph - 1));
}

void polyline(
    Image& im,
    const Axes& ax,
    const std::vector<double>& x,
    const std::vector<double>& y,
    RGB c
) {
  const Clip
      clip(im, ax.px + 1, ax.py + 1, ax.px + ax.pw - 1, ax.py + ax.ph - 1);
  for (size_t i = 1; i < std::min(x.size(), y.size()); ++i) {
    line(im, sx(ax, x[i - 1]), sy(ax, y[i - 1]), sx(ax, x[i]), sy(ax, y[i]), c);
  }
}

void marker(
    Image& im,
    const Axes& ax,
    double x,
    double y,
    RGB c,
    int r
) {
  const Clip
      clip(im, ax.px + 1, ax.py + 1, ax.px + ax.pw - 1, ax.py + ax.ph - 1);
  const int cx = sx(ax, x), cy = sy(ax, y);
  for (int j = -r; j <= r; j++)
    for (int i = -r; i <= r; i++)
      if (i * i + j * j <= r * r) put(im, cx + i, cy + j, c);
}

}

namespace {

constexpr double SETTLE_S = 2.0;
constexpr int BINS = 21;
constexpr double MIN_COMPLETION = 95.0;
constexpr double MIN_ROTATE = 99.0;
constexpr double MIN_DIRECTION = 95.0;
constexpr double DEADZONE_MIN_SPEED = 0.025;
constexpr double SMALL_LO = 0.05, SMALL_HI = 0.15;
constexpr double STILL_CMD = 0.02;
constexpr int DEADZONE_BINS = 50;
constexpr int GAP_GRID = 201;
constexpr size_t GAP_MIN_RUNS = 50;
constexpr double AXIS_LO = -0.1, AXIS_HI = 1.1;
constexpr int PLOT_X = 110, PLOT_Y = canvas::HEADER_H + canvas::STRIP_H;
constexpr int PLOT_SIZE = 620;
constexpr int SHEET_COLS = 3;
constexpr canvas::RGB RED = {214, 39, 40};
constexpr unsigned char DEADZONE_ALPHA = 64;

const std::vector<std::string> GAINS = {"1.0", "1.5", "2.0"};
const std::vector<std::string> MODES = {"rotate", "direction"};
const std::vector<std::string> ENGINES = {"mujoco", "physx"};

const std::map<std::string, canvas::RGB> ENGINE_INK = {
    {"mujoco", canvas::WHITE},
    {"physx", canvas::BLACK},
};

struct Run {
  double cmd_vx, cmd_vy, cmd_wz, vx, vy, wz;
};

struct Runs {
  std::vector<Run> complete;
  long fell = 0;
};

std::vector<std::string> split(const std::string& line) {
  std::vector<std::string> out;
  std::string cell;
  std::istringstream in(line);
  while (std::getline(in, cell, ',')) out.push_back(cell);
  if (!line.empty() && line.back() == ',') out.emplace_back();
  return out;
}

std::vector<fs::path> batches(
    const fs::path& folder,
    const std::string& policy,
    const std::string& engine
) {
  std::vector<fs::path> out;
  if (!fs::is_directory(folder)) return out;
  const std::regex name("r[0-9][0-9]\\." + policy + "\\." + engine + "\\.csv");
  for (const auto& entry : fs::directory_iterator(folder)) {
    if (std::regex_match(entry.path().filename().string(), name)) {
      out.push_back(entry.path());
    }
  }
  std::sort(out.begin(), out.end());
  return out;
}

Runs read_runs(const std::vector<fs::path>& paths) {
  Runs runs;
  for (const fs::path& path : paths) {
    std::ifstream in(path);
    std::string line;
    if (!std::getline(in, line)) continue;
    const std::vector<std::string> head = split(line);
    auto col = [&](const std::string& n) {
      const auto it = std::find(head.begin(), head.end(), n);
      if (it == head.end()) {
        throw std::runtime_error(path.string() + " has no column " + n);
      }
      return size_t(it - head.begin());
    };
    const size_t outcome = col("outcome");
    const size_t c[6] = {
        col("cmd_vx_mps"),
        col("cmd_vy_mps"),
        col("cmd_wz_rads"),
        col("vx_mps"),
        col("vy_mps"),
        col("wz_rads")
    };
    while (std::getline(in, line)) {
      if (line.empty()) continue;
      const std::vector<std::string> f = split(line);
      if (f[outcome] != "complete") {
        ++runs.fell;
        continue;
      }
      double v[6];
      for (int k = 0; k < 6; ++k) v[k] = std::stod(f[c[k]]);
      runs.complete.push_back({v[0], v[1], v[2], v[3], v[4], v[5]});
    }
  }
  return runs;
}

double percent_complete(const std::vector<fs::path>& paths) {
  long done = 0, total = 0;
  for (const fs::path& path : paths) {
    std::ifstream in(path);
    std::string line;
    if (!std::getline(in, line)) continue;
    const std::vector<std::string> head = split(line);
    const size_t outcome =
        size_t(std::find(head.begin(), head.end(), "outcome") - head.begin());
    while (std::getline(in, line)) {
      if (line.empty()) continue;
      ++total;
      done += split(line)[outcome] == "complete";
    }
  }
  return total == 0 ? -1.0 : 100.0 * double(done) / double(total);
}

double mode_completion(
    const fs::path& results,
    const std::string& policy,
    const std::string& mode,
    const std::string& gain
) {
  double sum = 0.0;
  for (const std::string& engine : ENGINES) {
    const double p = percent_complete(
        batches(results / (mode + "_gainx" + gain), policy, engine)
    );
    if (p < 0.0) return 0.0;
    sum += p;
  }
  return sum / double(ENGINES.size());
}

double completion(
    const fs::path& results,
    const std::string& policy
) {
  double sum = 0.0;
  for (const std::string& mode : MODES) {
    sum += mode_completion(results, policy, mode, "1.0");
  }
  return sum / double(MODES.size());
}

void bench_xy(
    const std::string& mode,
    const std::vector<Run>& runs,
    std::vector<double>& x,
    std::vector<double>& y
) {
  for (const Run& r : runs) {
    if (mode == "rotate") {
      x.push_back(std::fabs(r.cmd_wz));
      y.push_back(std::fabs(r.wz));
    } else {
      x.push_back(std::hypot(r.cmd_vx, r.cmd_vy));
      y.push_back(std::hypot(r.vx, r.vy));
    }
  }
}

double quantile(
    std::vector<double> v,
    double q
) {
  std::sort(v.begin(), v.end());
  const double at = q * double(v.size() - 1);
  const size_t lo = size_t(std::floor(at));
  const size_t hi = std::min(lo + 1, v.size() - 1);
  return v[lo] + (v[hi] - v[lo]) * (at - double(lo));
}

int bin_of(
    double x,
    int bins
) {
  return std::clamp(int(std::floor(x * bins)), 0, bins - 1);
}

struct Binned {
  std::vector<double> centre, median, q1, q3;
};

Binned binned(
    const std::vector<double>& x,
    const std::vector<double>& y,
    int bins
) {
  std::vector<std::vector<double>> bx(static_cast<size_t>(bins)),
      by(static_cast<size_t>(bins));
  for (size_t i = 0; i < x.size(); ++i) {
    const int b = bin_of(x[i], bins);
    bx[size_t(b)].push_back(x[i]);
    by[size_t(b)].push_back(y[i]);
  }
  Binned out;
  for (int b = 0; b < bins; ++b) {
    if (bx[size_t(b)].empty()) continue;
    out.centre.push_back(quantile(bx[size_t(b)], 0.5));
    out.median.push_back(quantile(by[size_t(b)], 0.5));
    out.q1.push_back(quantile(by[size_t(b)], 0.25));
    out.q3.push_back(quantile(by[size_t(b)], 0.75));
  }
  return out;
}

struct Deadzone {
  double command, speed;
};

std::optional<Deadzone> deadzone(
    const std::vector<double>& x,
    const std::vector<double>& y
) {
  std::vector<std::vector<double>> bx(DEADZONE_BINS), by(DEADZONE_BINS);
  for (size_t i = 0; i < x.size(); ++i) {
    const int b = bin_of(x[i], DEADZONE_BINS);
    bx[size_t(b)].push_back(x[i]);
    by[size_t(b)].push_back(y[i]);
  }
  for (int b = 0; b < DEADZONE_BINS; ++b) {
    if (bx[size_t(b)].empty()) continue;
    const double speed = quantile(by[size_t(b)], 0.5);
    if (speed >= DEADZONE_MIN_SPEED) {
      return Deadzone{quantile(bx[size_t(b)], 0.5), speed};
    }
  }
  return std::nullopt;
}

std::string fmt(
    const char* f,
    double a,
    double b = 0.0,
    double c = 0.0
) {
  char buf[256];
  std::snprintf(buf, sizeof(buf), f, a, b, c);
  return buf;
}

std::string upper(std::string s) {
  for (char& c : s) c = char(std::toupper(static_cast<unsigned char>(c)));
  return s;
}

canvas::Image blank() {
  canvas::Image im;
  im.px.assign(size_t(canvas::TILE) * canvas::TILE * 3, 0);
  canvas::fill_rect(im, 0, 0, canvas::TILE, canvas::TILE, canvas::BG);
  canvas::rect(im, 0, 0, canvas::TILE, canvas::TILE, canvas::WHITE);
  canvas::rect(im, 1, 1, canvas::TILE - 2, canvas::TILE - 2, canvas::WHITE);
  return im;
}

void header(
    canvas::Image& im,
    const std::string& title,
    const std::string& caption,
    const std::string& value,
    const std::string& detail
) {
  const int x = 16 + canvas::text_width(title, canvas::TITLE_SCALE) + 16;
  canvas::text(im, 16, 12, title, canvas::WHITE, canvas::TITLE_SCALE);
  canvas::text(im, x, 12, caption, canvas::WHITE, canvas::TITLE_SCALE);
  canvas::text(im, 16, 44, value, canvas::WHITE, canvas::TITLE_SCALE);
  canvas::text(
      im,
      16 + canvas::text_width(value, canvas::TITLE_SCALE) + 16,
      44,
      detail,
      canvas::WHITE,
      canvas::TITLE_SCALE
  );
}

void dashed(
    canvas::Image& im,
    const canvas::Axes& ax,
    double x0,
    double y0,
    double x1,
    double y1,
    int on,
    int off
) {
  const canvas::Clip
      clip(im, ax.px + 1, ax.py + 1, ax.px + ax.pw - 1, ax.py + ax.ph - 1);
  const int a = canvas::sx(ax, x0), b = canvas::sy(ax, y0);
  const int dx = canvas::sx(ax, x1) - a, dy = canvas::sy(ax, y1) - b;
  const int n = std::max(std::abs(dx), std::abs(dy));
  for (int i = 0; i <= n; ++i) {
    if (i % (on + off) >= on) continue;
    const int x = a + int(std::lround(double(dx) * i / n));
    const int y = b + int(std::lround(double(dy) * i / n));
    canvas::put(im, x, y, canvas::WHITE);
  }
}

canvas::Axes plot_box(
    canvas::Image& im,
    const std::string& unit
) {
  canvas::Axes ax;
  ax.px = PLOT_X;
  ax.py = PLOT_Y;
  ax.pw = ax.ph = PLOT_SIZE;
  ax.x0 = ax.y0 = AXIS_LO;
  ax.x1 = ax.y1 = AXIS_HI;
  canvas::fill_rect(im, ax.px, ax.py, ax.pw, ax.ph, canvas::BG);
  canvas::rect(im, ax.px, ax.py, ax.pw, ax.ph, canvas::WHITE);
  const int ts = canvas::LABEL_SCALE;
  for (int i = int(std::ceil(AXIS_LO * 10)); i <= int(AXIS_HI * 10); ++i) {
    const double v = 0.1 * i;
    const int x = canvas::sx(ax, v), y = canvas::sy(ax, v);
    for (int k = ax.py; k < ax.py + ax.ph; k += 4) {
      canvas::put(im, x, k, canvas::WHITE);
    }
    for (int k = ax.px; k < ax.px + ax.pw; k += 4) {
      canvas::put(im, k, y, canvas::WHITE);
    }
    if (i < 0 || i % 2) continue;
    for (int k = 0; k < 4; ++k) {
      canvas::put(im, x, ax.py + ax.ph + k, canvas::WHITE);
      canvas::put(im, ax.px - 1 - k, y, canvas::WHITE);
    }
    const std::string t = fmt("%.1f", v);
    canvas::text(
        im,
        x - canvas::text_width(t, ts) / 2,
        ax.py + ax.ph + 1,
        t,
        canvas::WHITE,
        ts
    );
    canvas::text(
        im,
        ax.px - 1 - canvas::text_width(t, ts),
        y - canvas::text_height(ts) / 2,
        t,
        canvas::WHITE,
        ts
    );
  }
  dashed(im, ax, AXIS_LO, AXIS_LO, AXIS_HI, AXIS_HI, 2, 4);
  const std::string xlabel = "COMMANDED (" + unit + ")";
  const std::string ylabel = "MEASURED (" + unit + ")";
  canvas::text(
      im,
      ax.px + (ax.pw - canvas::text_width(xlabel, ts)) / 2,
      ax.py + ax.ph + 1 + canvas::text_height(ts),
      xlabel,
      canvas::WHITE,
      ts
  );
  canvas::text_rot(
      im,
      ax.px - canvas::text_width("-0.1", ts) - canvas::text_height(ts) - 2,
      ax.py + (ax.ph + canvas::text_width(ylabel, ts)) / 2,
      ylabel,
      canvas::WHITE,
      ts
  );
  return ax;
}

void draw_engine(
    canvas::Image& im,
    const canvas::Axes& ax,
    const std::vector<double>& x,
    const std::vector<double>& y,
    canvas::RGB ink
) {
  const Binned b = binned(x, y, BINS);
  if (b.centre.empty()) return;
  {
    const canvas::Clip
        clip(im, ax.px + 1, ax.py + 1, ax.px + ax.pw - 1, ax.py + ax.ph - 1);
    for (size_t i = 0; i < b.centre.size(); ++i) {
      const int cx =
          canvas::sx(ax, b.centre[i]) + (ink.r == canvas::WHITE.r ? 1 : -1);
      const int lo = canvas::sy(ax, b.q1[i]), hi = canvas::sy(ax, b.q3[i]);
      canvas::line(im, cx, lo, cx, hi, ink);
      canvas::line(im, cx - 3, lo, cx + 3, lo, ink);
      canvas::line(im, cx - 3, hi, cx + 3, hi, ink);
    }
  }
  for (int k = 0; k < 4; ++k) {
    canvas::Axes thick = ax;
    thick.px += k % 2;
    thick.py += k / 2;
    canvas::polyline(im, thick, b.centre, b.median, ink);
  }
  for (size_t i = 0; i < b.centre.size(); ++i) {
    canvas::marker(im, ax, b.centre[i], b.median[i], ink, 6);
  }
}

void deadzone_box(
    canvas::Image& im,
    const canvas::Axes& ax,
    const Deadzone& dz
) {
  const canvas::Clip
      clip(im, ax.px + 1, ax.py + 1, ax.px + ax.pw - 1, ax.py + ax.ph - 1);
  const int x0 = canvas::sx(ax, 0.0), y1 = canvas::sy(ax, 0.0);
  const int x1 = canvas::sx(ax, dz.command), y0 = canvas::sy(ax, dz.speed);
  for (int y = y0; y <= y1; ++y) {
    for (int x = x0; x <= x1; ++x) canvas::blend(im, x, y, DEADZONE_ALPHA, RED);
  }
}

void lines_right(
    canvas::Image& im,
    const canvas::Axes& ax,
    const std::vector<std::string>& lines
) {
  const int s = canvas::LABEL_SCALE, lh = canvas::text_height(s) + 6;
  int w = 0;
  for (const std::string& line : lines) {
    w = std::max(w, canvas::text_width(line, s));
  }
  const int x = ax.px + ax.pw - 12 - w - 8;
  const int y = ax.py + ax.ph - 12 - lh * int(lines.size()) - 6;
  canvas::fill_rect(im, x, y, w + 16, lh * int(lines.size()) + 10, canvas::BG);
  canvas::rect(im, x, y, w + 16, lh * int(lines.size()) + 10, canvas::WHITE);
  for (size_t k = 0; k < lines.size(); ++k) {
    canvas::text(
        im,
        ax.px + ax.pw - 12 - canvas::text_width(lines[k], s),
        ax.py + ax.ph - 12 - lh * int(lines.size() - k),
        lines[k],
        canvas::WHITE,
        s
    );
  }
}

void lines_centre(
    canvas::Image& im,
    const canvas::Axes& ax,
    const std::vector<std::string>& lines
) {
  const int s = canvas::LABEL_SCALE, lh = canvas::text_height(s) + 10;
  const int top = ax.py + (ax.ph - lh * int(lines.size())) / 2;
  for (size_t k = 0; k < lines.size(); ++k) {
    const int w = canvas::text_width(lines[k], s);
    canvas::fill_rect(
        im,
        ax.px + (ax.pw - w) / 2 - 6,
        top + lh * int(k) - 3,
        w + 12,
        canvas::text_height(s) + 6,
        canvas::BG
    );
    canvas::text(
        im,
        ax.px + (ax.pw - w) / 2,
        top + lh * int(k),
        lines[k],
        canvas::WHITE,
        s
    );
  }
}

enum class Swatch { DOT_WHITE, DOT_BLACK, DOTTED, BOX, WHISKER };

void key(
    canvas::Image& im,
    const canvas::Axes& ax,
    const std::vector<std::pair<
        Swatch,
        std::string>>& items
) {
  const int s = canvas::LABEL_SCALE, lh = canvas::text_height(s) + 8;
  int w = 0;
  for (const auto& item : items) {
    w = std::max(w, canvas::text_width(item.second, s));
  }
  w += 44;
  const int h = int(items.size()) * lh + 8;
  const int x = ax.px + 8, y = ax.py + 8;
  canvas::fill_rect(im, x, y, w, h, canvas::BG);
  canvas::rect(im, x, y, w, h, canvas::WHITE);
  for (size_t i = 0; i < items.size(); ++i) {
    const int ly = y + 4 + int(i) * lh;
    const int cy = ly + canvas::text_height(s) / 2;
    switch (items[i].first) {
      case Swatch::DOT_WHITE:
      case Swatch::DOT_BLACK: {
        const canvas::RGB ink =
            items[i].first == Swatch::DOT_WHITE ? canvas::WHITE : canvas::BLACK;
        canvas::fill_rect(im, x + 6, cy, 22, 2, ink);
        for (int j = -6; j <= 6; ++j) {
          for (int k = -6; k <= 6; ++k) {
            if (j * j + k * k <= 36) canvas::put(im, x + 17 + k, cy + j, ink);
          }
        }
        break;
      }
      case Swatch::DOTTED:
        for (int k = 6; k < 28; k += 6) {
          canvas::fill_rect(im, x + k, cy, 2, 1, canvas::WHITE);
        }
        break;
      case Swatch::BOX:
        for (int j = cy - 6; j <= cy + 6; ++j) {
          for (int k = x + 8; k <= x + 26; ++k) {
            canvas::blend(im, k, j, DEADZONE_ALPHA, RED);
          }
        }
        break;
      case Swatch::WHISKER:
        canvas::fill_rect(im, x + 17, cy - 7, 1, 15, canvas::WHITE);
        canvas::fill_rect(im, x + 14, cy - 7, 7, 1, canvas::WHITE);
        canvas::fill_rect(im, x + 14, cy + 7, 7, 1, canvas::WHITE);
        break;
    }
    canvas::text(im, x + 36, ly, items[i].second, canvas::WHITE, s);
  }
}

struct Panel {
  std::string mode, gain;
};

canvas::Image mode_tile(
    const std::string& policy,
    const fs::path& results,
    const std::string& mode,
    const std::string& gain,
    const std::pair<
        bool,
        std::string>& accepted,
    std::vector<Panel>& panels
) {
  const std::string unit = mode == "rotate" ? "rad/s" : "m/s";
  const std::string caption = "LEGS KP/KD X" + gain;
  canvas::Image im = blank();
  canvas::text(
      im,
      PLOT_X,
      canvas::HEADER_H + 2,
      mode == "rotate"
          ? "MEASURED YAW RATE VS COMMAND, 2-10 S, FALLEN RUNS EXCLUDED"
          : "MEASURED PLANAR SPEED VS COMMAND, 2-10 S, FALLEN RUNS EXCLUDED",
      canvas::WHITE,
      canvas::LABEL_SCALE
  );
  const canvas::Axes ax = plot_box(im, unit);
  const auto& [ok, why] = accepted;
  if (!ok) {
    header(im, upper(mode), caption, "NOT ACCEPTED", "TOO MANY RUNS FALL");
    std::vector<std::string> lines;
    std::istringstream reasons(why);
    for (std::string reason; std::getline(reasons, reason, ';');) {
      lines.push_back(upper(reason.substr(reason.find_first_not_of(' '))));
    }
    lines_centre(im, ax, lines);
    return im;
  }
  std::vector<std::string> notes;
  struct Trace {
    std::vector<double> x, y;
    canvas::RGB ink;
  };
  std::vector<Trace> traces;
  for (size_t e = ENGINES.size(); e-- > 0;) {
    const std::string& engine = ENGINES[e];
    const canvas::RGB ink = ENGINE_INK.at(engine);
    const Runs runs =
        read_runs(batches(results / (mode + "_gainx" + gain), policy, engine));
    std::vector<double> x, y;
    bench_xy(mode, runs.complete, x, y);
    std::vector<std::string> note = {
        upper(engine) + " " + std::to_string(runs.complete.size()) + " OK / " +
        std::to_string(runs.fell) + " FELL"
    };
    if (!x.empty()) {
      traces.push_back({x, y, ink});
      const std::optional<Deadzone> dz = deadzone(x, y);
      if (dz) deadzone_box(im, ax, *dz);
      note.push_back(
          upper(engine) + " DEADZONE " +
          (dz ? fmt("%.3f, FIRST SPEED %.3f ", dz->command, dz->speed) + unit
              : std::string("NONE"))
      );
    }
    notes.insert(notes.begin(), note.begin(), note.end());
  }
  for (const Trace& t : traces) draw_engine(im, ax, t.x, t.y, t.ink);
  if (traces.empty()) {
    header(im, upper(mode), caption, "NO RUNS", "");
    return im;
  }
  panels.push_back({mode, gain});
  header(im, upper(mode), caption, "", "");
  key(im,
      ax,
      {{Swatch::DOT_WHITE, "MUJOCO, MEDIAN PER BIN"},
       {Swatch::DOT_BLACK, "PHYSX, MEDIAN PER BIN"},
       {Swatch::WHISKER, "QUARTILES PER BIN"},
       {Swatch::DOTTED, "MEASURED = COMMANDED"},
       {Swatch::BOX, "DEADZONE PER ENGINE"}});
  lines_right(im, ax, notes);
  return im;
}

struct Precision {
  double error, walk_right, walk_delivered, turn_right, turn_delivered;
  std::map<std::string, double> drift;
};

double median(std::vector<double> v) { return quantile(v, 0.5); }

Precision precision(
    const fs::path& results,
    const std::string& policy,
    const std::string& gain
) {
  Precision pr{};
  std::vector<double> walk_along, walk_ratio, walk_error;
  std::vector<double> turn_along, turn_ratio, turn_error;
  for (const std::string& engine : ENGINES) {
    const Runs walk = read_runs(
        batches(results / ("direction_gainx" + gain), policy, engine)
    );
    std::vector<double> still;
    for (const Run& r : walk.complete) {
      const double c = std::hypot(r.cmd_vx, r.cmd_vy);
      if (c < STILL_CMD) still.push_back(std::hypot(r.vx, r.vy));
      if (c < SMALL_LO || c >= SMALL_HI) continue;
      const double along = (r.vx * r.cmd_vx + r.vy * r.cmd_vy) / c;
      walk_along.push_back(along);
      walk_ratio.push_back(along / c);
      walk_error.push_back(std::hypot(r.vx - r.cmd_vx, r.vy - r.cmd_vy) / c);
    }
    pr.drift[engine] = still.empty() ? 0.0 : median(still);
    const Runs turn =
        read_runs(batches(results / ("rotate_gainx" + gain), policy, engine));
    for (const Run& r : turn.complete) {
      const double c = std::fabs(r.cmd_wz);
      if (c < SMALL_LO || c >= SMALL_HI) continue;
      const double along = r.wz * std::copysign(1.0, r.cmd_wz);
      turn_along.push_back(along);
      turn_ratio.push_back(along / c);
      turn_error.push_back(
          (std::fabs(r.wz - r.cmd_wz) + std::hypot(r.vx, r.vy)) / c
      );
    }
  }
  auto share = [](const std::vector<double>& v) {
    if (v.empty()) return 0.0;
    return double(std::count_if(
               v.begin(),
               v.end(),
               [](double a) { return a >= DEADZONE_MIN_SPEED; }
           )) /
           double(v.size());
  };
  pr.walk_right = share(walk_along);
  pr.turn_right = share(turn_along);
  pr.walk_delivered = walk_ratio.empty() ? 0.0 : median(walk_ratio);
  pr.turn_delivered = turn_ratio.empty() ? 0.0 : median(turn_ratio);
  pr.error = 0.5 * ((walk_error.empty() ? 1.0 : median(walk_error)) +
                    (turn_error.empty() ? 1.0 : median(turn_error)));
  return pr;
}

struct Score {
  std::string policy, gain;
  Precision pr;
};

void write_sheet(
    const std::vector<canvas::Image>& tiles,
    int cols,
    const fs::path& path
) {
  const int rows = (int(tiles.size()) + cols - 1) / cols;
  const int W = cols * canvas::TILE, H = rows * canvas::TILE;
  std::vector<unsigned char> px(size_t(W) * H * 3);
  for (size_t i = 0; i + 2 < px.size(); i += 3) {
    px[i] = canvas::BG.r;
    px[i + 1] = canvas::BG.g;
    px[i + 2] = canvas::BG.b;
  }
  for (size_t k = 0; k < tiles.size(); ++k) {
    const int cx = int(k % size_t(cols)) * canvas::TILE;
    const int cy = int(k / size_t(cols)) * canvas::TILE;
    for (int y = 0; y < canvas::TILE; ++y) {
      std::memcpy(
          &px[(size_t(cy + y) * W + cx) * 3],
          &tiles[k].px[size_t(y) * canvas::TILE * 3],
          size_t(canvas::TILE) * 3
      );
    }
  }
  if (!stbi_write_png(path.c_str(), W, H, 3, px.data(), W * 3)) {
    throw std::runtime_error("cannot write " + path.string());
  }
}

std::vector<double> yaw_curve(const std::vector<fs::path>& paths) {
  const Runs runs = read_runs(paths);
  if (runs.complete.size() < GAP_MIN_RUNS) return {};
  std::vector<double> x, y;
  bench_xy("rotate", runs.complete, x, y);
  const Binned b = binned(x, y, BINS);
  std::vector<double> curve(GAP_GRID);
  for (int i = 0; i < GAP_GRID; ++i) {
    const double g = double(i) / double(GAP_GRID - 1);
    if (g <= b.centre.front()) {
      curve[size_t(i)] = b.median.front();
    } else if (g >= b.centre.back()) {
      curve[size_t(i)] = b.median.back();
    } else {
      const size_t k = size_t(
          std::upper_bound(b.centre.begin(), b.centre.end(), g) -
          b.centre.begin()
      );
      const double t = (g - b.centre[k - 1]) / (b.centre[k] - b.centre[k - 1]);
      curve[size_t(i)] = b.median[k - 1] + t * (b.median[k] - b.median[k - 1]);
    }
  }
  return curve;
}

std::vector<std::pair<
    double,
    std::string>>
yaw_gaps(
    const fs::path& results,
    const std::vector<std::string>& policies
) {
  const fs::path folder = results / "rotate_gainx1.0";
  std::vector<std::pair<double, std::string>> gaps;
  std::printf("yaw-rate gap, mujoco vs physx, legs KP/KD x1.0\n");
  for (const std::string& policy : policies) {
    if (mode_completion(results, policy, "rotate", "1.0") <= MIN_ROTATE)
      continue;
    if (mode_completion(results, policy, "direction", "1.0") <= MIN_DIRECTION) {
      continue;
    }
    const std::vector<double> a = yaw_curve(batches(folder, policy, "mujoco"));
    const std::vector<double> b = yaw_curve(batches(folder, policy, "physx"));
    if (a.empty() || b.empty()) continue;
    double gap = 0.0;
    const double h = 1.0 / double(GAP_GRID - 1);
    for (int i = 1; i < GAP_GRID; ++i) {
      gap += 0.5 * h *
             (std::fabs(a[size_t(i)] - b[size_t(i)]) +
              std::fabs(a[size_t(i - 1)] - b[size_t(i - 1)]));
    }
    gaps.emplace_back(gap, policy);
  }
  std::sort(gaps.begin(), gaps.end());
  double sum = 0.0;
  for (const auto& [gap, policy] : gaps) {
    std::printf("  %-26s %.4f\n", policy.c_str(), gap);
    sum += gap;
  }
  if (!gaps.empty()) {
    std::printf("  %-26s %.4f\n", "average", sum / double(gaps.size()));
  }
  return gaps;
}

std::vector<std::string> discover(const fs::path& results) {
  std::set<std::string> found;
  const std::regex name("r[0-9][0-9]\\.(.+)\\.(mujoco|physx)\\.csv");
  for (const std::string& mode : MODES) {
    for (const std::string& gain : GAINS) {
      const fs::path folder = results / (mode + "_gainx" + gain);
      if (!fs::is_directory(folder)) continue;
      for (const auto& entry : fs::directory_iterator(folder)) {
        std::smatch m;
        const std::string file = entry.path().filename().string();
        if (std::regex_match(file, m, name)) found.insert(m[1]);
      }
    }
  }
  return {found.begin(), found.end()};
}

}

std::map<
    std::string,
    double>
punch_survival(const fs::path& folder) {
  std::map<std::string, std::pair<double, long>> sum;
  if (!fs::is_directory(folder)) return {};
  const std::regex name(
      "tier_[ab]\\.r[0-9][0-9]\\.(.+)\\.(mujoco|physx)\\.csv"
  );
  for (const auto& entry : fs::directory_iterator(folder)) {
    std::smatch m;
    const std::string file = entry.path().filename().string();
    if (!std::regex_match(file, m, name)) continue;
    std::ifstream in(entry.path());
    std::string line;
    if (!std::getline(in, line)) continue;
    const std::vector<std::string> head = split(line);
    const size_t at = size_t(
        std::find(head.begin(), head.end(), "survival_s") - head.begin()
    );
    auto& [total, runs] = sum[m[1]];
    while (std::getline(in, line)) {
      if (line.empty()) continue;
      total += std::stod(split(line)[at]);
      ++runs;
    }
  }
  std::map<std::string, double> mean;
  for (const auto& [policy, tr] : sum) {
    if (tr.second > 0) mean[policy] = tr.first / double(tr.second);
  }
  return mean;
}

int main(
    int argc,
    char** argv
) {
  fs::path results = "results";
  fs::path out = "results/cmd_vs_real_movements";
  fs::path readme = "README.md";
  std::set<std::string> only;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--results" && i + 1 < argc) {
      results = argv[++i];
    } else if (a == "--out" && i + 1 < argc) {
      out = argv[++i];
    } else if (a == "--readme" && i + 1 < argc) {
      readme = argv[++i];
    } else if (a == "--no-readme") {
      readme.clear();
    } else if (a == "--policy" && i + 1 < argc) {
      only.insert(argv[++i]);
    } else {
      std::fprintf(
          stderr,
          "usage: %s [--results DIR] [--out DIR] [--readme FILE | "
          "--no-readme]\n"
          "          [--policy NAME]...\n"
          "  one chart per policy: rotate and direction rows, legs KP/KD "
          "x1.0/x1.5/x2.0 columns,\n"
          "  for policies whose x1.0 completion averages > %g%%; a gain is "
          "drawn only when\n"
          "  rotate completes > %g%% and direction > %g%% at it\n",
          argv[0],
          MIN_COMPLETION,
          MIN_ROTATE,
          MIN_DIRECTION
      );
      return a == "--help" || a == "-h" ? 0 : 2;
    }
  }

  std::vector<std::string> policies;
  for (const std::string& policy : discover(results)) {
    if (!only.empty() && !only.count(policy)) continue;
    if (completion(results, policy) <= MIN_COMPLETION) continue;
    policies.push_back(policy);
  }
  fs::create_directories(out);
  for (const auto& entry : fs::directory_iterator(out)) {
    const fs::path ext = entry.path().extension();
    if (ext == ".png" || ext == ".jpg") fs::remove(entry.path());
  }
  std::map<std::string, std::vector<canvas::Image>> sheets;
  std::vector<Score> scores;
  for (const std::string& policy : policies) {
    std::map<std::string, std::pair<bool, std::string>> accepted;
    for (const std::string& gain : GAINS) {
      const double rot = mode_completion(results, policy, "rotate", gain);
      const double dir = mode_completion(results, policy, "direction", gain);
      std::string why;
      if (rot <= MIN_ROTATE) {
        why += fmt("rotate completes %.1f%%, needs > %g%%", rot, MIN_ROTATE);
      }
      if (dir <= MIN_DIRECTION) {
        if (!why.empty()) why += "; ";
        why +=
            fmt("direction completes %.1f%%, needs > %g%%", dir, MIN_DIRECTION);
      }
      accepted[gain] = {why.empty(), why};
    }
    if (std::none_of(accepted.begin(), accepted.end(), [](const auto& g) {
          return g.second.first;
        })) {
      std::printf("skip %s: no legs KP/KD gain accepted\n", policy.c_str());
      continue;
    }
    std::vector<Panel> panels;
    std::vector<canvas::Image>& tiles = sheets[policy];
    for (const std::string& mode : MODES) {
      for (const std::string& gain : GAINS) {
        tiles.push_back(
            mode_tile(policy, results, mode, gain, accepted[gain], panels)
        );
      }
    }
    for (const std::string& gain : GAINS) {
      if (std::none_of(panels.begin(), panels.end(), [&](const Panel& p) {
            return p.gain == gain;
          })) {
        continue;
      }
      scores.push_back({policy, gain, precision(results, policy, gain)});
      break;
    }
  }
  std::stable_sort(
      scores.begin(),
      scores.end(),
      [](const Score& a, const Score& b) { return a.pr.error < b.pr.error; }
  );
  std::printf(
      "%-5s %-26s %-5s %6s %10s %10s %15s %10s %10s\n",
      "",
      "policy",
      "gain",
      "error",
      "walk right",
      "walk deliv",
      "drift mj/px",
      "turn right",
      "turn deliv"
  );
  const std::string head =
      "| # | policy | legs<br>KP/KD | error | walk:<br>right<br>way | "
      "walk:<br>delivered | drift<br>at zero m/s<br>MuJoCo / PhysX | "
      "turn:<br>right<br>way | turn:<br>delivered |\n"
      "|---:|---|---:|---:|---:|---:|---:|---:|---:|\n";
  const std::string metrics =
      "Precision is scored on small commands, " + fmt("%g", SMALL_LO) + " to " +
      fmt("%g", SMALL_HI) +
      " m/s or rad/s, by movement along the command, both engines pooled, "
      "at the first accepted gain (x1.0, else x1.5, else x2.0). Error is the "
      "median distance between measured and commanded velocity relative to "
      "the command, walk and turn averaged (in rotate the planar drift counts "
      "as error): 0% is perfect, 100% is what standing still scores, above "
      "100% the robot moves the wrong way. Right way is the share of runs "
      "moving at least " +
      fmt("%g", DEADZONE_MIN_SPEED) +
      " along the command; delivered is the median fraction of the command "
      "achieved along it; drift at zero is the median planar speed for "
      "direction commands under " +
      fmt("%g", STILL_CMD) + " m/s. Rows are ordered by error.\n\n";
  std::string table = head;
  auto pct = [](double v) { return fmt("%.0f%%", std::round(v * 100.0)); };
  for (size_t i = 0; i < scores.size(); ++i) {
    const Score& sc = scores[i];
    const Precision& pr = sc.pr;
    char prefix[8];
    std::snprintf(prefix, sizeof(prefix), "%03zu_", i % 1000);
    std::vector<canvas::Image>& tiles = sheets.at(sc.policy);
    write_sheet(tiles, SHEET_COLS, out / (prefix + sc.policy + ".png"));
    tiles.clear();
    const std::string values =
        " | x" + sc.gain + " | " + pct(pr.error) + " | " + pct(pr.walk_right) +
        " | " + fmt("%.2f", std::max(pr.walk_delivered, 0.0)) + " | " +
        fmt("%.3f", pr.drift.at("mujoco")) + " / " +
        fmt("%.3f", pr.drift.at("physx")) + " | " + pct(pr.turn_right) + " | " +
        fmt("%.2f", std::max(pr.turn_delivered, 0.0)) + " |\n";
    table += "| " + std::to_string(i) + " | [`" + sc.policy + "`](" +
             (out / (prefix + sc.policy + ".png")).generic_string() + ")" +
             values;
    std::printf(
        "%-5s %-26s x%-4s %6s %10s %10.2f %7.3f/%7.3f %10s %10.2f\n",
        prefix,
        sc.policy.c_str(),
        sc.gain.c_str(),
        pct(pr.error).c_str(),
        pct(pr.walk_right).c_str(),
        pr.walk_delivered,
        pr.drift.at("mujoco"),
        pr.drift.at("physx"),
        pct(pr.turn_right).c_str(),
        pr.turn_delivered
    );
  }
  const auto gaps = yaw_gaps(results, policies);
  std::string gap_table = "| policy | gap |\n|---|---:|\n";
  double gap_sum = 0.0;
  for (const auto& [gap, policy] : gaps) {
    gap_table += "| `" + policy + "` | " + fmt("%.4f", gap) + " |\n";
    gap_sum += gap;
  }
  if (!gaps.empty()) {
    gap_table +=
        "| average | " + fmt("%.4f", gap_sum / double(gaps.size())) + " |\n";
  }
  if (!readme.empty()) {
    const std::string begin = "<!-- cmd_vs_real_movements:begin -->";
    const std::string end = "<!-- cmd_vs_real_movements:end -->";
    std::ifstream in(readme);
    const std::string text(
        (std::istreambuf_iterator<char>(in)),
        std::istreambuf_iterator<char>()
    );
    const size_t b = text.find(begin), e = text.find(end);
    if (b == std::string::npos || e == std::string::npos || e < b) {
      std::fprintf(
          stderr,
          "cmd_vs_real_movements: %s has no %s ... %s markers\n",
          readme.string().c_str(),
          begin.c_str(),
          end.c_str()
      );
      return 1;
    }
    double gap_mean = 0.0;
    for (const auto& g : gaps) gap_mean += g.first / double(gaps.size());
    std::string best;
    if (!scores.empty()) {
      const Score& top = scores.front();
      const fs::path chart = out / ("000_" + top.policy + ".png");
      const std::map<std::string, double> surv =
          punch_survival(results / "punches_gainx1.0");
      std::vector<std::pair<double, std::string>> order;
      for (const auto& [p, v] : surv) order.emplace_back(-v, p);
      std::sort(order.begin(), order.end());
      std::string punches;
      for (size_t k = 0; k < order.size(); ++k) {
        if (order[k].second != top.policy) continue;
        punches = "In the punch tour above it survives " +
                  fmt("%.1f", -order[k].first) + " s on average, " +
                  std::to_string(k + 1) + " of " +
                  std::to_string(order.size()) + " (the best, `" +
                  order.front().second + "`, " +
                  fmt("%.1f", -order.front().first) + " s).";
      }
      const Precision& pr = top.pr;
      best = "![" + top.policy + "](" + chart.generic_string() + ")\n\n`" +
             top.policy + "` ranks first at legs KP/KD x" + top.gain +
             ", with " + pct(pr.error) +
             " error on small commands. It moves the commanded way in " +
             pct(pr.walk_right) + " of small walking commands and " +
             pct(pr.turn_right) + " of small turns, delivers " +
             fmt("%.0f%%", pr.walk_delivered * 100.0) + " and " +
             fmt("%.0f%%", pr.turn_delivered * 100.0) +
             " of them, and drifts " + fmt("%.3f", pr.drift.at("mujoco")) +
             " / " + fmt("%.3f", pr.drift.at("physx")) +
             " m/s (MuJoCo / PhysX) when asked to stand still. " + punches +
             "\n";
    }
    const std::string section =
        begin + "\n" +
        "Each run holds one command for 10 s, with no punches: a yaw rate in "
        "`--mode rotate`, a planar velocity in `--mode direction`, both drawn "
        "uniformly per run id up to 1 rad/s and 1 m/s. "
        "`./benchmark.sh --benchmark rotate|direction` runs every policy at "
        "legs "
        "KP/KD x1.0, x1.5 and x2.0 for 4096 run ids each into "
        "`results/<mode>_gainx<G>/`, and `build/cmd_vs_real_movements` charts "
        "measured against commanded speed into `" +
        out.generic_string() + "/`. " + metrics +
        "Policies are listed when their legs KP/KD x1.0 runs complete > " +
        fmt("%g", MIN_COMPLETION) +
        "% on average and at least one gain completes > " +
        fmt("%g", MIN_ROTATE) + "% in rotate and > " +
        fmt("%g", MIN_DIRECTION) + "% in direction.\n\n" + table + "\n" + best +
        "\nMuJoCo and PhysX differ in yaw-rate tracking by " +
        fmt("%.3f", gap_mean) +
        " on average (area between the engines' median curves at x1.0), from " +
        (gaps.empty() ? std::string("-") : "`" + gaps.front().second + "`") +
        " closest to " +
        (gaps.empty() ? std::string("-") : "`" + gaps.back().second + "`") +
        " furthest:\n\n" + gap_table + end;
    std::ofstream(readme) << text.substr(0, b) + section +
                                 text.substr(e + end.size());
    std::printf("updated %s\n", readme.string().c_str());
  }
  return 0;
}
