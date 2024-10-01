#!/usr/bin/env bash

# Set user account and run values
USER_NAME=${USER_NAME:-wineuser}
USER_UID=${USER_UID:-1010}
USER_GID=${USER_GID:-"${USER_UID}"}
USER_HOME=${USER_HOME:-/home/"${USER_NAME}"}

# Create the user account
grep -q ":${USER_GID}:$" /etc/group || groupadd --gid "${USER_GID}" "${USER_NAME}"
grep -q "^${USER_NAME}:" /etc/passwd || useradd --shell /bin/bash --uid "${USER_UID}" --gid "${USER_GID}" --no-create-home --home-dir "${USER_HOME}" "${USER_NAME}"

# Create the user's home if it doesn't exist
[ -d "${USER_HOME}" ] || mkdir -p "${USER_HOME}"

# Configure timezone
ln -snf "/usr/share/zoneinfo/${TZ}" /etc/localtime
echo "${TZ}" > /etc/timezone

# Run in X11 redirection mode (default) or with xvfb

    # Run xvfb
#    if is_enabled "${USE_XVFB}"; then
#        nohup /usr/bin/Xvfb "${XVFB_SERVER}" -screen "${XVFB_SCREEN}" "${XVFB_RESOLUTION}" >/dev/null 2>&1 &
#    fi

    # Generate .Xauthority using xauth with .Xkey sourced from host
if [ -f /root/.Xkey ];
then
	[ -f /root/.Xauthority ] || touch /root/.Xauthority
	xauth add "$DISPLAY" . "$(cat /root/.Xkey)"
fi

    # Run in X11 redirection mode as $USER_NAME (default)

        # Copy and take ownership of .Xauthority for X11 redirection
if [ -f /root/.Xauthority ];
then # && is_disabled "${USE_XVFB}"; then
	cp /root/.Xauthority "${USER_HOME}"
	chown "${USER_UID}":"${USER_GID}" "${USER_HOME}/.Xauthority"
fi

        # Run in X11 redirection mode as
# exec gosu "${USER_NAME}" "$@"
