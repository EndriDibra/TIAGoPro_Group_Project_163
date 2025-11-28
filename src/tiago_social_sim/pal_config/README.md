# PAL User Configuration Overrides

This directory contains PAL configuration override files that are automatically deployed to `/home/user/.pal/config/` when the Docker container starts.

## Files

### 99_user_costmap.yaml
Reduces costmap update frequencies to lower computational load:
- `local_costmap.update_frequency`: 2.0 (down from 10.0)
- `local_costmap.publish_frequency`: 1.0 (down from 10.0)
- `global_costmap.update_frequency`: 0.5 (down from 5.0)
- `expected_update_rate`: 1.0 (more lenient)

### 99_user_laser.yaml
Reduces laser scan resolution to lower computational load:
- `angle_increment`: 0.012 radians (doubled from 0.0058, reducing scan points by ~50%)

## How it works

1. **Storage**: These files are stored in `src/tiago_social_sim/pal_config/`
2. **Deployment**: The Docker entrypoint script (`docker/entrypoint.sh`) automatically copies these files to `/home/user/.pal/config/` on container startup
3. **Precedence**: PAL's configuration system gives these files the highest precedence, overriding system defaults

## Customization

To modify these parameters:
1. Edit the YAML files in this directory
2. Rebuild the package: `colcon build --packages-select tiago_social_sim`
3. Restart the container to apply changes

## Reverting

To disable these overrides:
- Delete or rename the files in this directory
- Rebuild and restart the container
