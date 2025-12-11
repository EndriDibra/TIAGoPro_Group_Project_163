FROM development-tiago-pro-34:alum-25.01

# Install PyTorch and torchvision first
RUN pip install --no-cache-dir \
    torch>=2.0.0 \
    torchvision>=0.15.0

# Force software rendering for simulation stability
ENV LIBGL_ALWAYS_SOFTWARE=1

# Install missing build dependencies
USER root
RUN apt-get update && apt-get install -y \
    ros-humble-launch \
    ros-humble-launch-ros \
    ros-humble-ament-cmake \
    ros-humble-robot-localization \
    ros-humble-image-pipeline \
    && rm -rf /var/lib/apt/lists/*


# Copy and install Python requirements
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir \
    -r /tmp/requirements.txt && rm /tmp/requirements.txt

# Copy entrypoint script
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

USER user
WORKDIR /home/user

# Add build alias to .bashrc for convenience
RUN echo "alias build='colcon build --symlink-install && source install/setup.bash'" >> /home/user/.bashrc
RUN echo "alias start_sim='ros2 launch tiago_social_sim simulation.launch.py navigation:=True slam:=True world_name:=simple_office &> ~/src/tmp/output.log'" >> /home/user/.bashrc
RUN echo "alias src='source install/setup.bash'" >> /home/user/.bashrc
RUN echo "alias run_experiments='nohup ~/src/run_experiments.sh &> ~/src/tmp/experiment_suite.log &'" >> /home/user/.bashrc

# Set the entrypoint
ENTRYPOINT ["/entrypoint.sh"]
CMD ["/bin/bash"]