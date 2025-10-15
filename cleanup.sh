# #!/bin/bash
# # cleanup_unnecessary_files.sh
# # Run this from the project root: /home/suriya/cassie_rl_walking-master

# echo "🧹 Starting cleanup of unnecessary files..."
# echo "Files to keep: visualize_harpy_play.py and cassie_env/"
# echo ""

# Function to safely remove with confirmation
safe_remove() {
    if [ -e "$1" ]; then
        echo "✓ Removing: $1"
        rm -rf "$1"
    fi
}

# # 1. Remove accidental pip install artifacts
# echo "📦 Removing pip artifacts..."
# safe_remove "=1.0.0"
# safe_remove "=2.12.0"

# # 2. Remove MuJoCo archive
# echo "🗜️ Removing MuJoCo archive..."
# safe_remove "mujoco210-linux-x86_64.tar.gz"

# # 3. Remove Python cache directories
# echo "🐍 Removing __pycache__ directories..."
# safe_remove "__pycache__"
# safe_remove "configs/__pycache__"
# safe_remove "ppo/__pycache__"
# safe_remove "rlenv/__pycache__"
# safe_remove "utility/__pycache__"

# # 4. Remove build artifacts
# echo "🔨 Removing build artifacts..."
# safe_remove "cassie_rl_walking.egg-info"

# # 5. Remove log files
# echo "📝 Removing log files..."
# safe_remove "MUJOCO_LOG.TXT"
# safe_remove "MJDATA.TXT"
# safe_remove "exe/MUJOCO_LOG.TXT"

# # 6. Remove image files
# echo "🖼️ Removing images..."
# safe_remove "screenshot.png"
# safe_remove "Figure_1.png"

# # 7. Remove duplicate viewer scripts (keeping visualize_harpy_play.py)
# echo "👁️ Removing duplicate viewer/visualization scripts..."
# safe_remove "harpy_3d_viewer.py"
# safe_remove "harpy_mujoco_images.py"
# safe_remove "view_harpy_3d_fixed.py"
# safe_remove "view_harpy_3d.py"
# safe_remove "view_harpy_complete.py"
# safe_remove "view_harpy_simple_3d.py"
# safe_remove "view_harpy_sophisticated.py"
# safe_remove "simple_harpy_view.py"
# safe_remove "simple_harpy_visualization.py"
# safe_remove "mujoco_harpy_viewer.py"
# safe_remove "visualize_harpy_precise.py"
# safe_remove "visualize_harpy_sketch.py"
# safe_remove "visualize_harpy_standing.py"

# # 8. Remove test files
# echo "🧪 Removing test files..."
# safe_remove "test_harpy_cassie_legs.py"
# safe_remove "test_harpy_cassie_precise.py"
# safe_remove "test_harpy_standing.py"
# safe_remove "test_hopping_reference.py"
# safe_remove "test_mujoco_viewer.py"

# # 9. Remove temporary analysis/documentation files
# echo "📄 Removing temporary documentation..."
# safe_remove "BLDC_HARMONIC_DRIVE_ANALYSIS.md"
# safe_remove "BLDC_HARMONIC_IMPLEMENTATION_SUMMARY.md"
# safe_remove "THRUSTER_ANALYSIS.md"
# safe_remove "THRUSTER_SNAPSHOT_SUMMARY.md"
# safe_remove "THRUSTER_STABILIZATION_ANALYSIS.md"
# safe_remove "UPDATE_SUMMARY.md"
# safe_remove "thruster_design_diagram.txt"
# safe_remove "thruster_stabilization_diagram.txt"

# # 10. Remove empty directories
# echo "📁 Removing empty directories..."
# safe_remove "results"
# safe_remove "external/baselines"
# safe_remove "external"

# # 11. Remove old analysis plots
# echo "📊 Removing old plots..."
# safe_remove "PLOTS"

# # 12. Remove old/experimental scripts
# echo "🔧 Removing old scripts..."
# safe_remove "train_modern.py"  # You have scripts/train.py
# safe_remove "monitor_training.py"
# safe_remove "analyze_training_progress.sh"

# # 13. Remove old checkpoint directories (keep only versatile_walking)
# echo "💾 Cleaning up old checkpoints (keeping versatile_walking)..."
# safe_remove "ckpts/leonardo_hopping_rnds42"
# safe_remove "ckpts/modern_training_rnds42"
# safe_remove "ckpts/new_training_walking_with_thrusters_rnds1"
# safe_remove "ckpts/new_training_walking_without_thrusters_rnds1"
# safe_remove "ckpts/new_training_walking_without_thrusters_rnds1_cont1"

# # 14. Remove old log directories
# echo "📋 Cleaning up old logs..."
# safe_remove "logs/leonardo_hopping_rnds42"
# safe_remove "logs/leonardo_hybrid_rnds42"
# safe_remove "logs/modern_training_rnds42"
# safe_remove "logs/new_training_walking_with_thrusters_rnds1"
# safe_remove "logs/new_training_walking_without_thrusters_rnds1"
# safe_remove "logs/new_training_walking_without_thrusters_rnds1_cont1"
# safe_remove "logs/test_cassie_rnds42"

# echo ""
# echo "✅ Cleanup complete!"
# echo ""
# echo "📌 Files kept as requested:"
# echo "  - cassie_env/ (virtual environment)"
# echo "  - visualize_harpy_play.py"
# echo ""
# echo "🔍 To verify, run: git status"

echo "🐳 Removing Docker-related files..."
safe_remove ".dockerignore"
safe_remove "Dockerfile"
safe_remove "docker-compose.yml"
safe_remove "docker-compose.yaml"
safe_remove ".docker"