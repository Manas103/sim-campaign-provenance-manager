#include "approach_comparison/types.hpp"

#include <fstream>
#include <sstream>

namespace approach_comparison {

std::vector<FeatureRow> read_corpus_tsv(std::istream& is) {
    std::vector<FeatureRow> rows;
    std::string line;
    std::getline(is, line); // header
    while (std::getline(is, line)) {
        if (line.empty()) continue;
        std::istringstream ss(line);
        FeatureRow r;
        std::string field;
        std::getline(ss, field, '\t');
        r.sample_id = std::stoi(field);
        std::getline(ss, r.true_label, '\t');
        std::getline(ss, field, '\t');
        r.distance_m = std::stod(field);
        std::getline(ss, field, '\t');
        r.heading_diff_deg = std::stod(field);
        std::getline(ss, field, '\t');
        r.relative_speed_mps = std::stod(field);
        std::getline(ss, field, '\t');
        r.decel_mps2 = std::stod(field);
        std::getline(ss, field, '\t');
        r.time_gap_s = std::stod(field);
        rows.push_back(r);
    }
    return rows;
}

std::vector<FeatureRow> read_corpus_tsv_file(const std::string& path) {
    std::ifstream f(path);
    return read_corpus_tsv(f);
}

void write_predictions_tsv(std::ostream& os, const std::vector<Prediction>& predictions) {
    os << "sample_id\tpredicted_label\n";
    for (const auto& p : predictions) os << p.sample_id << '\t' << p.predicted_label << '\n';
}

std::vector<Prediction> read_predictions_tsv_file(const std::string& path) {
    std::ifstream f(path);
    std::vector<Prediction> out;
    std::string line;
    std::getline(f, line); // header
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::istringstream ss(line);
        Prediction p;
        std::string field;
        std::getline(ss, field, '\t');
        p.sample_id = std::stoi(field);
        std::getline(ss, p.predicted_label, '\t');
        out.push_back(p);
    }
    return out;
}

} // namespace approach_comparison
