#include <matplot/matplot.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <optional>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
namespace mp = matplot;

namespace {

constexpr double SETTLE_S = 2.0;
constexpr int BINS = 21;
constexpr double MIN_COMPLETION = 95.0;
constexpr double MIN_ROTATE = 99.0;
constexpr double MIN_DIRECTION = 95.0;
constexpr double DEADZONE_MIN_SPEED = 0.025;
constexpr int DEADZONE_BINS = 50;
constexpr int GAP_GRID = 201;
constexpr size_t GAP_MIN_RUNS = 50;
constexpr double AXIS_LO = -0.08, AXIS_HI = 1.08;

const std::vector<std::string> GAINS = {"1.0", "1.5", "2.0"};
const std::vector<std::string> MODES = {"rotate", "direction"};
const std::vector<std::string> ENGINES = {"mujoco", "physx"};

using Colour = std::array<float, 4>;
const std::map<std::string, Colour> ENGINE_COLOUR = {
    {"mujoco", {0.0f, 0.165f, 0.471f, 0.839f}},
    {"physx", {0.0f, 0.922f, 0.408f, 0.204f}},
};
const Colour INK = {0.0f, 0.043f, 0.043f, 0.043f};
const Colour INK_SECONDARY = {0.0f, 0.322f, 0.318f, 0.306f};
const Colour DEADZONE = {0.0f, 0.839f, 0.153f, 0.157f};

Colour transparent(
    Colour c,
    float alpha
) {
  c[0] = 1.0f - alpha;
  return c;
}

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

struct Fit {
  double slope, intercept, rmse, r2;
};

Fit fit_line(
    const std::vector<double>& x,
    const std::vector<double>& y
) {
  const double n = double(x.size());
  double sx = 0, sy = 0, sxx = 0, sxy = 0;
  for (size_t i = 0; i < x.size(); ++i) {
    sx += x[i];
    sy += y[i];
    sxx += x[i] * x[i];
    sxy += x[i] * y[i];
  }
  const double slope = (n * sxy - sx * sy) / (n * sxx - sx * sx);
  const double intercept = (sy - slope * sx) / n;
  double se = 0;
  for (size_t i = 0; i < x.size(); ++i) {
    const double e = y[i] - (slope * x[i] + intercept);
    se += e * e;
  }
  const double mean = sy / n;
  double st = 0;
  for (const double v : y) st += (v - mean) * (v - mean);
  return {slope, intercept, std::sqrt(se / n), st > 0.0 ? 1.0 - se / st : 0.0};
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

void draw_engine(
    const mp::axes_handle& ax,
    const std::vector<double>& x,
    const std::vector<double>& y,
    const std::string& engine
) {
  const Binned b = binned(x, y, BINS);
  if (b.centre.empty()) return;
  std::vector<double> px = b.centre, py = b.q1;
  for (size_t i = b.centre.size(); i-- > 0;) {
    px.push_back(b.centre[i]);
    py.push_back(b.q3[i]);
  }
  mp::fill(ax, px, py)->color(transparent(ENGINE_COLOUR.at(engine), 0.18f));
  auto line = mp::plot(ax, b.centre, b.median, "-o");
  line->line_width(2).marker_size(4).color(ENGINE_COLOUR.at(engine));
}

void reference(const mp::axes_handle& ax) {
  auto diagonal = mp::plot(
      ax,
      std::vector<double>{0.0, 1.0},
      std::vector<double>{0.0, 1.0},
      "--"
  );
  diagonal->color(INK_SECONDARY).line_width(1);
}

void note(
    const mp::axes_handle& ax,
    double x,
    double y,
    const std::string& s,
    const Colour& c,
    float size
) {
  auto t = mp::text(ax, x, y, s);
  t->color(c).font_size(size);
}

struct Panel {
  std::string mode, gain;
  double r2, slope;
  std::vector<double> deadzones;
};

std::vector<Panel> grid_chart(
    const std::string& policy,
    const fs::path& results,
    const fs::path& out,
    const std::map<
        std::string,
        std::pair<
            bool,
            std::string>>& accepted
) {
  std::vector<Panel> panels;
  auto f = mp::figure(true);
  f->size(1650, 1150);
  std::string shown;
  for (const char ch : policy) {
    if (ch == '_') shown += "\\\\";
    shown += ch;
  }
  f->title(
      shown +
      ": commanded vs measured   (blue mujoco, orange physx, black "
      "line fit over both, red deadzone per engine)"
  );
  for (size_t r = 0; r < MODES.size(); ++r) {
    const std::string& mode = MODES[r];
    const std::string unit = mode == "rotate" ? "rad/s" : "m/s";
    for (size_t c = 0; c < GAINS.size(); ++c) {
      const std::string& gain = GAINS[c];
      auto ax = mp::subplot(
          f,
          int(MODES.size()),
          int(GAINS.size()),
          int(r * GAINS.size() + c)
      );
      mp::hold(ax, true);
      mp::title(ax, mode + " · legs KP/KD x" + gain);
      mp::xlabel(ax, "commanded (" + unit + ")");
      mp::ylabel(ax, "measured (" + unit + ")");
      mp::xlim(ax, {AXIS_LO, AXIS_HI});
      mp::ylim(ax, {AXIS_LO, AXIS_HI});
      mp::axis(ax, mp::square);
      reference(ax);
      const auto& [ok, why] = accepted.at(gain);
      if (!ok) {
        std::vector<std::string> lines = {"not accepted:"};
        std::istringstream reasons(why);
        for (std::string reason; std::getline(reasons, reason, ';');) {
          lines.push_back(reason.substr(reason.find_first_not_of(' ')));
        }
        lines.push_back("(runs falling within 10 s)");
        for (size_t k = 0; k < lines.size(); ++k) {
          note(
              ax,
              0.05,
              0.6 - 0.08 * double(k),
              lines[k],
              INK_SECONDARY,
              k ? 8 : 9
          );
        }
        continue;
      }
      std::vector<double> all_x, all_y, panel_zones;
      std::vector<std::string> counts, zones;
      for (const std::string& engine : ENGINES) {
        const Runs runs = read_runs(
            batches(results / (mode + "_gainx" + gain), policy, engine)
        );
        counts.push_back(
            engine + " " + std::to_string(runs.complete.size()) + " ok / " +
            std::to_string(runs.fell) + " fell"
        );
        std::vector<double> x, y;
        bench_xy(mode, runs.complete, x, y);
        if (x.empty()) {
          zones.push_back(engine + " deadzone: no runs");
          continue;
        }
        draw_engine(ax, x, y, engine);
        const std::optional<Deadzone> dz = deadzone(x, y);
        panel_zones.push_back(dz ? dz->command : 1.0);
        if (dz) {
          auto rect = mp::rectangle(ax, 0.0, 0.0, dz->command, dz->speed);
          rect->fill(true).color(transparent(DEADZONE, 0.2f));
          zones.push_back(
              engine + " deadzone " + fmt("%.3f", dz->command) +
              ", first speed " + fmt("%.3f", dz->speed) + " " + unit
          );
        } else {
          zones.push_back(engine + " deadzone: none");
        }
        all_x.insert(all_x.end(), x.begin(), x.end());
        all_y.insert(all_y.end(), y.begin(), y.end());
      }
      if (!all_x.empty()) {
        const Fit fit = fit_line(all_x, all_y);
        panels.push_back({mode, gain, fit.r2, fit.slope, panel_zones});
        auto line = mp::plot(
            ax,
            std::vector<double>{0.0, 1.0},
            std::vector<double>{fit.intercept, fit.slope + fit.intercept}
        );
        line->color(INK).line_width(1.5);
        note(
            ax,
            0.0,
            1.0,
            fmt("fit: slope %.2f, R^2 %.2f, RMSE %.3f",
                fit.slope,
                fit.r2,
                fit.rmse),
            INK,
            9
        );
      }
      for (size_t k = 0; k < zones.size(); ++k) {
        note(ax, 0.0, 0.92 - 0.07 * double(k), zones[k], DEADZONE, 8);
      }
      for (size_t k = 0; k < counts.size(); ++k) {
        note(ax, 0.45, 0.08 - 0.07 * double(k), counts[k], INK_SECONDARY, 7);
      }
    }
  }
  mp::save(f, out.string());
  return panels;
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
  struct Score {
    std::string policy, gain;
    double r2, deadzone;
    std::map<std::string, double> mode_r2, mode_deadzone, mode_slope;
    int rank_walk = 0, rank_rotate = 0, rank_walk_zone = 0,
        rank_rotate_zone = 0;
    int rank_sum() const {
      return rank_walk + rank_rotate + rank_walk_zone + rank_rotate_zone;
    }
  };
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
    const fs::path file = out / (policy + ".png");
    const std::vector<Panel> panels =
        grid_chart(policy, results, file, accepted);
    for (const std::string& gain : GAINS) {
      double r2 = 0.0, zone = 0.0;
      int n_r2 = 0, n_zone = 0;
      std::map<std::string, double> mode_r2, mode_zone, mode_slope;
      for (const Panel& p : panels) {
        if (p.gain != gain) continue;
        r2 += p.r2;
        ++n_r2;
        mode_r2[p.mode] = p.r2;
        mode_slope[p.mode] = p.slope;
        double panel_zone = 0.0;
        for (const double z : p.deadzones) {
          zone += z;
          ++n_zone;
          panel_zone += z;
        }
        mode_zone[p.mode] = panel_zone / double(p.deadzones.size());
      }
      if (n_r2 == 0) continue;
      scores.push_back(
          {policy,
           gain,
           r2 / n_r2,
           zone / n_zone,
           mode_r2,
           mode_zone,
           mode_slope}
      );
      break;
    }
  }
  auto rank = [&](auto key, auto set) {
    std::vector<size_t> order(scores.size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
      return key(scores[a]) < key(scores[b]);
    });
    for (size_t r = 0; r < order.size(); ++r) set(scores[order[r]], int(r) + 1);
  };
  rank(
      [](const Score& s) { return -s.mode_r2.at("direction"); },
      [](Score& s, int r) { s.rank_walk = r; }
  );
  rank(
      [](const Score& s) { return -s.mode_r2.at("rotate"); },
      [](Score& s, int r) { s.rank_rotate = r; }
  );
  rank(
      [](const Score& s) { return s.mode_deadzone.at("direction"); },
      [](Score& s, int r) { s.rank_walk_zone = r; }
  );
  rank(
      [](const Score& s) { return s.mode_deadzone.at("rotate"); },
      [](Score& s, int r) { s.rank_rotate_zone = r; }
  );
  std::stable_sort(
      scores.begin(),
      scores.end(),
      [](const Score& a, const Score& b) {
        return a.rank_sum() != b.rank_sum() ? a.rank_sum() < b.rank_sum()
                                            : a.r2 > b.r2;
      }
  );
  std::printf(
      "%-5s %-26s %-5s %13s %13s %13s %13s %4s\n",
      "",
      "policy",
      "gain",
      "walk R2",
      "rotate R2",
      "walk zone",
      "rotate zone",
      "sum"
  );
  std::string table =
      "| # | policy | legs<br>KP/KD | "
      "linear<br>real<br>vs<br>commanded<br>walk<br>speed | "
      "linear<br>real<br>vs<br>commanded<br>rotation<br>speed | "
      "walk<br>speed<br>deadzone | rotation<br>speed<br>deadzone |\n"
      "|---:|---|---:|---:|---:|---:|---:|\n";
  std::string md =
      "# Commanded vs measured movement\n\n"
      "Generated by `build/cmd_vs_real_movements` from "
      "`results/{rotate,direction}_gainx*`.\n\n"
      "Each run holds one command for 10 s: a yaw rate in `rotate`, a planar "
      "velocity in `direction`. Measured rates are taken over t = 2 s to the "
      "end, runs that fell excluded. A policy is charted when its legs KP/KD "
      "x1.0 completion averages > " +
      fmt("%g", MIN_COMPLETION) +
      "% over both engines and benchmarks, and a gain is drawn when rotate "
      "completes > " +
      fmt("%g", MIN_ROTATE) + "% and direction > " + fmt("%g", MIN_DIRECTION) +
      "% at it.\n\n"
      "Ranking uses the first accepted gain (x1.0, else x1.5, else x2.0): "
      "linear real vs commanded walk speed (direction) and rotation speed "
      "(rotate) are R^2 of the line fitted to measured vs commanded with both "
      "engines pooled, the share of the measured speed's variance the line "
      "explains, higher is better; the walk and "
      "rotation deadzones are the command "
      "needed to reach a measured " +
      fmt("%g", DEADZONE_MIN_SPEED) +
      ", averaged over both engines (1.0 when never reached). Each of the four "
      "is ranked on its own, 1 = lowest, and policies are "
      "ordered by the sum of the four ranks.\n\n"
      "| # | policy | legs<br>KP/KD | "
      "linear<br>real<br>vs<br>commanded<br>walk<br>speed | "
      "linear<br>real<br>vs<br>commanded<br>rotation<br>speed | "
      "walk<br>speed<br>deadzone | rotation<br>speed<br>deadzone |\n"
      "|---:|---|---:|---:|---:|---:|---:|\n";
  auto linearity = [](double r2) {
    return fmt("%.0f%%", std::round(r2 * 100.0));
  };
  auto cell = [](double v, int r) {
    return fmt("%.4f", v) + " (" + std::to_string(r) + ")";
  };
  for (size_t i = 0; i < scores.size(); ++i) {
    const Score& sc = scores[i];
    char prefix[8];
    std::snprintf(prefix, sizeof(prefix), "%03zu_", i);
    fs::rename(out / (sc.policy + ".png"), out / (prefix + sc.policy + ".png"));
    const std::string values =
        " | x" + sc.gain + " | " + linearity(sc.mode_r2.at("direction")) +
        " | " + linearity(sc.mode_r2.at("rotate")) + " | " +
        fmt("%.4f", sc.mode_deadzone.at("direction")) + " | " +
        fmt("%.4f", sc.mode_deadzone.at("rotate")) + " |\n";
    table += "| " + std::to_string(i) + " | [`" + sc.policy + "`](" +
             (out / (prefix + sc.policy + ".png")).generic_string() + ")" +
             values;
    md += "| " + std::to_string(i) + " | [`" + sc.policy + "`](" + prefix +
          sc.policy + ".png)" + values;
    std::printf(
        "%-5s %-26s x%-4s %13s %13s %13s %13s %4d\n",
        prefix,
        sc.policy.c_str(),
        sc.gain.c_str(),
        cell(sc.mode_r2.at("direction"), sc.rank_walk).c_str(),
        cell(sc.mode_r2.at("rotate"), sc.rank_rotate).c_str(),
        cell(sc.mode_deadzone.at("direction"), sc.rank_walk_zone).c_str(),
        cell(sc.mode_deadzone.at("rotate"), sc.rank_rotate_zone).c_str(),
        sc.rank_sum()
    );
  }
  const auto gaps = yaw_gaps(results, policies);
  md +=
      "\n## Yaw-rate gap, MuJoCo vs PhysX, legs KP/KD x1.0\n\n"
      "Area between the two engines' median yaw-rate curves over commands "
      "0..1 rad/s.\n\n"
      "| policy | gap |\n|---|---:|\n";
  double gap_sum = 0.0;
  for (const auto& [gap, policy] : gaps) {
    md += "| `" + policy + "` | " + fmt("%.4f", gap) + " |\n";
    gap_sum += gap;
  }
  if (!gaps.empty()) {
    md += "| average | " + fmt("%.4f", gap_sum / double(gaps.size())) + " |\n";
  }
  std::ofstream(out / "README.md") << md;
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
        punches =
            "In the punch tour above it survives " +
            fmt("%.1f", -order[k].first) + " s on average, " +
            std::to_string(k + 1) + " of " + std::to_string(order.size()) +
            " (the best, `" + order.front().second + "`, " +
            fmt("%.1f", -order.front().first) +
            " s), so it is not the policy that stays up best under punches; "
            "it is the one whose speed follows the walking and rotation "
            "commands most cleanly.";
      }
      best =
          "![" + top.policy + "](" + chart.generic_string() + ")\n\n`" +
          top.policy + "` ranks first at legs KP/KD x" + top.gain +
          ". Its measured speed follows a straight line through the command "
          "in both benchmarks, R\u00b2 " +
          fmt("%.0f%%", std::round(top.mode_r2.at("direction") * 100.0)) +
          " walking and " +
          fmt("%.0f%%", std::round(top.mode_r2.at("rotate") * 100.0)) +
          " turning, and it starts moving almost at once, with deadzones of " +
          fmt("%.3f", top.mode_deadzone.at("direction")) + " m/s and " +
          fmt("%.3f", top.mode_deadzone.at("rotate")) +
          " rad/s. The line is not y = x: it walks and turns at about " +
          fmt("%.0f%%", top.mode_slope.at("direction") * 100.0) + " and " +
          fmt("%.0f%%", top.mode_slope.at("rotate") * 100.0) +
          " of the command, a constant gain a teleop mapping can scale away. " +
          punches + "\n";
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
        out.generic_string() +
        "/`. Linear real vs commanded is R^2 of the line fitted to measured vs "
        "commanded with both engines pooled, the share of the measured speed's "
        "variance the line explains, higher is better; deadzone is the "
        "command needed to reach a measured " +
        fmt("%g", DEADZONE_MIN_SPEED) +
        ", averaged over both engines. Policies are listed when their legs "
        "KP/KD x1.0 runs complete > " +
        fmt("%g", MIN_COMPLETION) +
        "% on average and at least one gain completes > " +
        fmt("%g", MIN_ROTATE) + "% in rotate and > " +
        fmt("%g", MIN_DIRECTION) +
        "% in direction; values are at the first such gain, and the order sums "
        "each column's rank, 1 = lowest.\n\n" +
        table + "\n" + best +
        "\nMuJoCo and PhysX differ in yaw-rate tracking by " +
        fmt("%.3f", gap_mean) +
        " on average (area between the engines' median curves at x1.0), from " +
        (gaps.empty() ? std::string("-") : "`" + gaps.front().second + "`") +
        " closest to " +
        (gaps.empty() ? std::string("-") : "`" + gaps.back().second + "`") +
        " furthest; per policy in [" + (out / "README.md").generic_string() +
        "](" + (out / "README.md").generic_string() + ").\n" + end;
    std::ofstream(readme) << text.substr(0, b) + section +
                                 text.substr(e + end.size());
    std::printf("updated %s\n", readme.string().c_str());
  }
  std::printf("wrote %s\n", (out / "README.md").string().c_str());
  return 0;
}
