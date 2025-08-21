function launchrun() {
    gpus="$1"
    logfile="$2"
    shift 2  # Remove the first two arguments (gpus and logfile)

    # Generate a semi-unique session ID using date
    session_id="session_$(date +"%Y%m%d%H%M%S")"

    # Create a tmux session, run the command inside it, and detach
    tmux new-session -d -s "$session_id" "CUDA_VISIBLE_DEVICES=$gpus $* > $logfile 2>&1"

    # Optionally, you can also rename the window (replace "mywindow" with your desired name)
    # tmux rename-window -t "$session_id:0" mywindow

    # Detach from the session
    tmux detach-client

    echo "Session ID: $session_id"
}
