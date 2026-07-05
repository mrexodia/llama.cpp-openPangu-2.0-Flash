#pragma once

#include <string>
#include <vector>

// Ref: https://huggingface.co/docs/hub/local-cache.md

namespace hf_cache {

struct hf_file {
    std::string path;
    std::string url;
    std::string local_path;
    std::string final_path;
    std::string oid;
    std::string repo_id;
};

using hf_files = std::vector<hf_file>;

struct hf_ref {
    std::string repo_id;
    std::string branch;
    std::string commit;
};

// Get files from HF API
// if ref is non-null, it receives the resolved branch/commit; pass it to
// update_ref() once all files are finalized
hf_files get_repo_files(
    const std::string & repo_id,
    const std::string & token,
    hf_ref * ref = nullptr
);

// Write refs/<branch>; call only after the snapshot is complete, otherwise
// the ref would point at a snapshot that does not exist on disk
void update_ref(const hf_ref & ref);

hf_files get_cached_files(const std::string & repo_id = {});

// Create snapshot path (link or move/copy) and return it
std::string finalize_file(const hf_file & file);

// Remove the entire cached directory for a repo, returns true if removed
bool remove_cached_repo(const std::string & repo_id);

} // namespace hf_cache
