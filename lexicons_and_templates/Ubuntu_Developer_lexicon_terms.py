TECH_LEXICON = {

    # Ubuntu-specific vocabulary: the product, its release cycle, and Ubuntu-branded
    # tooling. Release codenames/version numbers are especially high-signal since they
    # essentially only appear in Ubuntu discourse.
    "ubuntu_specific": [
        'ubuntu', 'kubuntu', 'gnu', 'ppa', 'ppas', 'personal package archive',
        'hardware drivers', 'restricted-drivers', 'envy', 'wubi', 'distrowatch',
        'update manager', 'upgrade', 'desktop iso', 'server iso', 'iso',
        'livecd', 'live cd', 'live usb', 'recovery cd', 'recovery cds',
        'unetbootin', 'software center', 'software sources', 'ubuntu one',
        'task version', 'backtrack', 'debian', 'redhat systems',
        'distro', 'automatix', 'netinstall', 'howto',
        # release codenames, Warty (4.10) through Wily (15.10)
        'warty', 'hoary', 'breezy', 'dapper', 'edgy', 'feisty', 'gutsy', 'hardy',
        'intrepid', 'jaunty', 'karmic', 'lucid', 'maverick', 'natty', 'oneiric',
        'precise', 'quantal', 'raring', 'saucy', 'trusty', 'utopic', 'vivid', 'wily',
        # release version numbers
        '4.10', '5.04', '5.10', '6.06', '6.10', '7.04', '7.10', '8.04', '8.10',
        '9.04', '9.10', '10.04', '10.10', '11.04', '11.10', '12.04', '12.10',
        '13.04', '13.10', '14.04', '14.10', '15.04', '15.10',
    ],

    # Package management: apt/dpkg ecosystem and the problems people hit with it.
    "package_management": [
        'dpkg', 'apt', 'apt-get', 'aptitude', 'apt-cache', 'apt-key',
        'apt-add-repository', 'add-apt-repository', 'dpkg-reconfigure',
        'sources.list', 'sources.list.d', 'package repository', 'software repository',
        'repositories', 'repos', 'repository', 'repo', 'deb package', '.deb file', 'deb', 'rpm',
        'package cache', 'broken package', 'broken packages', 'held package',
        'unmet dependencies', 'dependency problem', 'dependencies', 'dependency',
        'purge', 'autoremove', 'autoclean', 'apt-get update', 'apt-get upgrade',
        'apt-get dist-upgrade', 'synaptic', 'package manager', 'cache',
        'lesstif2-dev', 'lamp', 'lamp-server^', 'libapache2-mod-php', 'phpmyadmin',
        '.pkg', 'pkg file',
    ],

    # Core Linux system concepts, users/permissions, and process management.
    "linux_system": [
        'linux', 'kernel', 'root user', 'root account', 'root master', 'chown',
        'chmod', 'bash', 'sudo bash', 'sudo', 'sudo chroot', 'terminal',
        'command line', 'daemon', 'settings-daemon', 'pid', 'ps aux', 'ps',
        'pstree', 'dmesg', 'lspci', 'grep', 'killall', 'kill -9', 'fstab',
        'resolv.conf', 'xorg.conf', 'udev', 'init.d', 'upstart', 'sysvinit',
        'update-rc.d', 'chkconfig', 'useradd', 'user account', 'password',
        'i386', 'i586', 'k6', 'k7', 'pae', 'hyperthreading', 'core 2 duo',
        'config', 'gui', '.conf', '.inf', 'services', 'service', 'connection',
        'cutting edge', 'nv', 'vesa', 'sudo modprobe snd-emu10k1', 'syntax',
        'shell', 'xwindows', 'x windows', 'x window system',
        'acpi-support', 'pm-utils', 'pm_utils', 'computer', 'device', 'display',
    ],

    # Booting, GRUB, and the errors people hit getting the system to start.
    "boot": [
        'grub', 'grub2', 'grub rescue', 'grub menu', 'bootloader', 'boot loader',
        'dual boot', 'dualboot', 'boot partition', 'boot sector', 'mbr',
        'master boot record', 'initramfs', 'busybox', 'kernel panic',
        'recovery mode', 'failsafe mode', 'single user mode', 'black screen',
        'blank screen', 'boot screen', 'splash screen', 'splash',
        'no such partition', 'unknown filesystem', 'failed to boot',
        'cannot boot', "won't boot", 'os-prober', 'bootable', 'to boot', 'boot',
        'booting', 'gdm', 'lightdm', 'xdm', 'reboot',
    ],

    # Filesystems, disks, and directory paths.
    "filesystem": [
        'filesystem', 'file system', 'ntfs', 'ntfs partition', 'fat16', 'fat32',
        'vfat', 'fat filesystem', 'partition', 'mount', 'unmount', 'dismount',
        'automount', 'fdisk', 'gparted', 'testdisk', 'partedmagic', 'sata', 'hdd',
        'sdb', 'c drive', 'ext2', 'ext3', 'ext4', 'home directory', 'home folder',
        'root directory', 'root filesystem', 'working directory',
        'current directory', 'file path', '/home', '/usr', '/bin', '/sbin',
        '/etc', '/var', '/tmp', '/proc', '/dev', '/boot', '/mnt', '/media',
        '/var/log', 'read-only filesystem', 'read only filesystem',
        'permission denied', 'file permissions', 'file ownership', 'ownership',
        'uid', 'gid', 'symlink', 'symbolic link', 'hard link', 'directory',
        'local directory', 'staging directory', 'cdrom', 'flashdrive', 'usb',
        'md5', 'chkdsk run', 'mounted harddisk', 'block list',
        'filename', 'file name', 'disc', 'delete a file', 'mounted',
        'dvd', 'dvds',
    ],

    # Networking: connectivity, wifi, DNS/DHCP, and connection error phrases.
    "networking": [
        'dns', 'dhcp', 'static ip', 'ssid', 'wireless', 'wired connection', 'vpn',
        'vpn server', 'proxy', 'firewall', 'iptables', 'iptables-persistent',
        'ufw', 'nat', 'ip', 'hostname', 'hostnames', 'nameserver',
        'connection refused', 'connection timed out', 'connection time out',
        'disconnected', 'ethernet', 'wifi', 'wi-fi', 'wlan', 'wireless network',
        'network interface', 'network adapter', 'network card', 'nic', 'router',
        'access point', 'gateway', 'default gateway', 'subnet', 'subnet mask',
        'localhost', 'loopback', 'ping', 'packet', 'packet loss', 'latency',
        'bandwidth', 'port forwarding', 'port', 'ports', 'http', 'https', 'ftp',
        'sftp', 'telnet', 'ipv4', 'ipv6', 'networkmanager', 'network-manager',
        'wpa_supplicant', 'network unreachable', 'host unreachable',
        'connection reset', 'connection failed', 'cannot connect',
        'no internet connection', 'temporary failure resolving',
        'name resolution', 'failed to resolve', 'link-local', 'squid', 'postfix',
        'x server', 'ifdown/ifup', 'overlay', 'tunneling', 'modem', 'interfaces',
        'networks', 'bcm4318 driver', 'brcm80211', 'ndiswrapper', 'vps', 'vps hosts',
        'reactivate', 'gate way', 'url', 'lan cable', 'ethernet cable', 'hosts file',
        'ifconfig', 'address',
    ],

    # SSH and remote administration.
    "remote_access": [
        'ssh', 'openssh', 'ssh server', 'ssh client', 'ssh key', 'public key',
        'private key', 'authorized_keys', 'scp', 'rsync', 'remote server',
        'remote machine', 'remote desktop', 'vnc', 'x11 forwarding', 'putty',
        '.pem', '.pub',
    ],

    # Hardware, drivers, and vendor-specific gear.
    "hardware": [
        'memtest', 'ghz', 'mb', 'gb', 'tb', 'processor', 'motherboard', 'mobo',
        'radeon', 'nvidia', 'video card', 'soundcard', 'soundcards', 'sound', 'volume',
        'keyboard', 'trackpad', 'wacom', 'bios', 'overheating', 'laptop',
        'thinkpad', 'powerbook', 'macbook', 'mac', 'mac osx', 'netbook screen',
        'resolution', 'high resolution', 'low resolution', 'driver', 'fglrx',
        'aticonfig', 'jockey', 'lexmark', 'plugged in', 'alt', 'f2', 'ctrl',
        'backspace', 'solid state', 'pc', 'personal computer', 'devices',
        'displays', 'dell', 'apple', 'microsoft', 'ibm', 'adobe', 'azure',
        'azureus', 'ati', 'intel', 'amd', 'power management', 'pcm',
        'graphix', 'graphics', 'graphic', 'acer', 'card', 'cards', 'videocard', 'realtek', 's3',
        'pcmcia', 'bsod', 'mhz', 'sd', 'speed',
        # explicit plural compound -- 'macbook' alone can't catch this glued
        # form (same \b-boundary limitation as 'windows7'); see the
        # 2026-08-19 review.
        'macbooks',
    ],

    # Desktop environments and window/UI chrome.
    "desktop_ui": [
        'gnome', 'gnome 2', 'gnome shell', 'kde', 'xfce', 'lxde', 'mate',
        'desktop environment', 'window manager', 'metacity', 'compiz', 'xgl',
        'fluxbox', 'unity desktop', 'dash', 'launcher', 'panel', 'indicator',
        'system tray', 'notification area', 'workspace', 'desktop icon', 'gtk',
        'gtk2', 'gtk3', 'qt', 'gnome-terminal', 'konsole', 'xterm', 'screensaver',
        'visual effects', 'nautilus', 'system monitor', 'preferences dialog',
        'desktop', 'gdesklets',
    ],

    # Applications people were running / troubleshooting.
    "applications": [
        'firefox', 'chromium', 'safari', 'browser', 'opera', 'thunderbird',
        'evolution', 'pidgin', 'empathy', 'skype', 'ekiga', 'vlc', 'mplayer',
        'totem', 'rhythmbox', 'banshee', 'amarok', 'xmms', 'winamp', 'audacity',
        'pulseaudio', 'pulse', 'alsa', 'imgburn', 'poweriso', 'shotwell', 'gimp',
        'inkscape', 'openoffice', 'libreoffice', 'brasero', 'cheese', 'gedit',
        'nano', 'vi', 'vim', 'emacs', 'wireshark', 'virtualbox', 'vmware', 'wine',
        'google earth', 'quickbooks', 'outlook', 'mail client', 'pst file',
        'mailing list', 'deluge', 'torrent', 'peers', 'jar', 'wikipedia',
        'app/application', 'app', 'application', 'video player', 'player',
        'downloads', 'download',
        'extracting', 'zip', 'doc', 'docx', 'jpg', 'png', 'pdf', 'gpg', 'apache',
        'php', 'audio cd ripper', 'libcssdvd', 'mozilla', 'gitbash', 'game',
        'quake', 'windows', 'windows 98', 'xp', 'xp machine', 'vista',
        'vista machine',
        # concatenated version compounds -- 'windows' alone can't catch these
        # since \b fails at a letter/digit boundary with no separator
        # ("windows7", "windowsXP", ...); see the 2026-08-19 lexicon review.
        'windows7', 'windows_7', 'windowsxp', 'windows95', 'windows98',
        'windowsvista', 'windowsme', 'windowsnt', 'windows2000',
        'mencoder', 'flash plugin', 'flashplugin', 'adobe flash', 'beta',
        # concatenated Adobe Flash version compound, same reasoning as the
        # windows7-style compounds above.
        'flash64',
        'tumblr', 'gmail',
    ],

    # Common shell commands and general terminal actions.
    "commands": [
        'ls', 'cd', 'pwd', 'cp', 'mv', 'rm', 'mkdir', 'rmdir', 'cat', 'less',
        'more', 'head', 'tail', 'find', 'locate', 'which', 'whereis', 'echo',
        'export', 'env', 'man', 'whoami', 'passwd', 'top', 'htop', 'free', 'df',
        'du', 'uptime', 'strace', 'lsof', 'netstat', 'ss', 'dig', 'nslookup',
        'traceroute', 'tcpdump', 'curl', 'wget', 'nc', 'edit', 'print', 'return',
        'quit', 'reopen', 'copy/paste', 'copy', 'paste', 'scroll', 'scroll up',
        'scroll down',
        'automate', 'script', 'operator', 'unexpected operator', 'set up',
        'setup', 'enable', 'verified', 'burning files', 'burning', 'burn',
        'install', 'installation', 'installing', 'reinstall', 'reinstalled',
        'install.sh', 'options', 'submit', 'idle', 'installed', 'installer',
        'plugins', '64-bit', '64bit',
    ],

    # Developer / build tooling that predates or overlaps the 2004-2015 window.
    # (Kubernetes, kubectl, podman, and flatpak were deliberately left out as
    # anachronistic for an Ubuntu support corpus ending in 2015.)
    "dev_tools": [
        'git', 'clone', 'commit', 'push', 'pull', 'make', 'cmake', 'gcc', 'qemu',
        'kvm', 'vagrant', 'lxd', 'ansible', 'terraform', 'jenkins', 'docker',
        'npm', 'pip', 'virtualenv', 'cargo', 'nvm', 'snap', '.env', '.bashrc',
        '.zshrc', '.bash_login', 'compiled', 'compiler', 'binutils', 'executables',
        'uri', 'sql', 'mysql', 'postgresql', 'sqlite', 'library', 'libraries',
        'plugin', 'cloud', 'dataframe', 'design flaws', 'wildcard', 'wild card',
        'oss', 'dev', 'array', 'vector', 'ec2', 'scipy',
    ],

    # IRC/forum meta-vocabulary -- this is IRC support-channel discourse, so these
    # channel-mechanics terms are themselves useful technical-context signals.
    "irc_meta": [
        'nickserv', '/query', '/quit', 'offtopic', '!help', 'howtos',
        'expletives', 'chat', 'nopaste', 'pastebin', 'irssi', 'homepage',
        'web link', 'broken link', 'timeline', 'tutorial', '!trash',
        # explicit channel-name compound -- 'offtopic' alone can't catch this
        # glued form (same \b-boundary limitation as 'windows7'); see the
        # 2026-08-19 review.
        'ubuntuofftopic',
    ],

    # Problem/error phrasing -- the "genre markers" of a support request.
    "troubleshooting": [
        'not working', 'does not work', "won't work", 'unable to', 'cannot',
        "can't", 'failed to', 'error', 'error message', 'error code',
        'command not found', 'no such file', 'no such directory',
        'device not found', 'file not found', 'package not found', 'broken',
        'system freeze', 'system froze', 'freezes', 'lags/lag', 'lag', 'lags',
        'keeps crashing', 'keeps freezing', 'crash', 'crashing', 'crashed',
        'bugs', 'bug', 'file a bug', 'bug report', 'bug reports', 'fixes',
        'stability', 'reliability', 'data integrity', 'integrity',
        'unexpected operator', 'cutting edge',
    ],

    # Formerly split out as a "low_confidence" (weak-evidence-only) tier. That
    # distinction was retired per the 2026-08-19 lexicon review -- every term
    # below now matches at the same full confidence as every other category.
    # These are common enough in ordinary English that a match here carries a
    # higher false-positive rate than e.g. 'apt-get' or 'grub' would, but
    # that's now an accepted, deliberate tradeoff (favoring recall), not an
    # oversight.
    "general_computing": [
        'unity', 'java', 'oracle', 'ram', 'monitor', 'root', 'module',
        'permissions', 'package', 'program', 'software', 'version',
        'running version', 'channel', 'support', 'wiki', 'icon', 'server',
        'protocol', 'hosting', 'process', 'is running', 'default', 'admin',
        'machine', 'key', 'file', 'data', 'object', 'string', 'state',
        'state information', 'source', 'output', 'result', 'comment',
        'docstring', 'local', 'global', 'system-wide', 'list', 'page', 'pages',
        'size', 'line', 'line of code', 'tab', 'interface', 'account', 'site',
        'website', 'help', 'login', 'log', 'logs', 'logged', 'log in', 'log on',
        'log out', 'run', 'running', 'runs', 'none', 'update', 'user/username',
        'user', 'username', 'ported', 'window', 'format', 'convert', 'folder',
        'link', 'system', 'systems', 'command', 'resource independence',
        'updated', 'files',
    ],
}


def flatten_lexicon(lexicon=TECH_LEXICON):
    """Return every term as one flat, deduplicated list, in case a caller just
    wants a simple flat lexicon rather than the categorized structure."""
    terms = []
    seen = set()
    for category, words in lexicon.items():
        for word in words:
            key = word.lower()
            if key not in seen:
                seen.add(key)
                terms.append(word)
    return terms


# Flat list of every term across every category (as of the 2026-08-19 review,
# this includes what used to be the separately-gated 'low_confidence' terms --
# there is no more tier distinction to exclude).
LEXICON_TERMS = flatten_lexicon()


# ---------------------------------------------------------------------------
# Matching. TECH_LEXICON above is just data (a categorized "is this term
# present" wordlist); everything below turns it into a matcher, mirroring
# the sibling modules structural_patterns.py (regex "shapes") and
# sms_slang_lexicon.py (the slang/emoticon CSV) so all three lexicon
# components share the same matching conventions:
#   - re.escape() every term before it goes into a pattern (a few entries
#     here contain regex metacharacters: 'lamp-server^', '.bashrc', '.env',
#     '.conf', '.pem', 'sources.list', etc.)
#   - \b-bounded so multi-word phrases and ordinary terms don't fire as
#     substrings of unrelated words ('boot' must not match inside 'booting')
#   - case-insensitive, matching how sms_slang_lexicon.py now treats slang
# ---------------------------------------------------------------------------
import re

# TERM_CATEGORY: term (lowercased) -> category name.
# Built directly off TECH_LEXICON so it stays in sync with the source dict
# (rather than off the deduplicated/flattened LEXICON_TERMS list).
TERM_CATEGORY = {}
for _category, _words in TECH_LEXICON.items():
    for _word in _words:
        TERM_CATEGORY.setdefault(_word.lower(), _category)


def _token_pattern(term):
    """Build a \\b-equivalent boundary around `term` that also works for
    terms starting/ending in punctuation, where a literal \\b wouldn't fire
    (\\b requires a word/non-word transition; a term like '.bashrc' starts
    with a non-word char, so \\b at that edge only matches if the preceding
    context char is a word char -- which is backwards from what we want:
    preceded by whitespace, i.e. non-word, is the common case and \\b would
    reject it there). (?<!\\w) / (?!\\w) instead just require "not adjacent
    to a word character," which is what we actually mean at a punctuation edge.

    Note this is still a hard word-character boundary either way, so a term
    glued directly onto an adjacent word or digit with no separator at all
    ("windows7", "ubuntucomputer") will NOT match -- that's a known, accepted
    limitation (see the 2026-08-19 lexicon review), not a bug; loosening it
    would reopen substring false-positives ('grep' inside 'telegraph', etc.)
    that \\b exists to prevent. Known high-volume compounds are instead
    listed as their own explicit terms (e.g. 'windows7', 'videocard').
    """
    escaped = re.escape(term)
    start = r'\b' if re.match(r'\w', term[0]) else r'(?<!\w)'
    end = r'\b' if re.match(r'\w', term[-1]) else r'(?!\w)'
    return f'{start}{escaped}{end}'


def _build_term_matcher(terms):
    if not terms:
        return None
    ordered = sorted(set(terms), key=len, reverse=True)
    alt = '|'.join(_token_pattern(t) for t in ordered)
    return re.compile(f'(?i:{alt})')


TECH_MATCHER = _build_term_matcher(LEXICON_TERMS)


def find_tech_matches(text):
    """Scan `text` and return a list of (matched_text, category) tuples.

    As of the 2026-08-19 lexicon review, there is no more separate
    'low_confidence' tier -- every category (including the terms formerly
    held out as weak evidence, now under 'general_computing') matches at the
    same confidence level. Callers that used to pass
    include_low_confidence=True/False or call find_low_confidence_tech_matches
    separately should just call this function; both of those are gone."""
    if TECH_MATCHER is None:
        return []
    return [(m.group(0), TERM_CATEGORY.get(m.group(0).lower())) for m in TECH_MATCHER.finditer(text)]