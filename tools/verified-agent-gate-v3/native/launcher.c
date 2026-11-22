#define _GNU_SOURCE
#include "seccomp_policy.h"
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <linux/close_range.h>
#include <linux/memfd.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/prctl.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

static volatile sig_atomic_t interrupted = 0;
static void on_signal(int sig) { interrupted = sig; }
static void die(const char *message) {
    fprintf(stderr, "gate-sandbox: %s: %s\n", message, strerror(errno));
    exit(125);
}
static void reject(const char *message) {
    fprintf(stderr, "gate-sandbox: %s\n", message);
    exit(125);
}
static unsigned long number(const char *value, unsigned long lower, unsigned long upper) {
    char *end = NULL;
    errno = 0;
    unsigned long n = strtoul(value, &end, 10);
    if (errno || !*value || *end || n < lower || n > upper) reject("invalid numeric limit");
    return n;
}
static int is_within(const char *child, const char *parent) {
    size_t n = strlen(parent);
    return !strcmp(child, parent) || (!strncmp(child, parent, n) && child[n] == '/');
}
static int directory(const char *path, char resolved[PATH_MAX]) {
    if (!path || path[0] != '/') reject("job and output paths must be absolute");
    if (!realpath(path, resolved)) die("resolve directory");
    if (!strcmp(resolved, "/")) reject("filesystem root cannot be a job or output directory");
    int fd = open(resolved, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
    if (fd < 0) die("open directory");
    struct stat st;
    if (fstat(fd, &st)) die("stat directory");
    if (st.st_uid != getuid() || (st.st_mode & 0022)) reject("job and output directories must be owned by this user and not group/world writable");
    return fd;
}
static void limit(int resource, rlim_t amount) {
    struct rlimit value = { amount, amount };
    if (setrlimit(resource, &value)) die("set resource limit");
}
static double monotonic_seconds(void) {
    struct timespec now;
    if (clock_gettime(CLOCK_MONOTONIC, &now)) die("read monotonic clock");
    return (double)now.tv_sec + (double)now.tv_nsec / 1000000000.0;
}
static int filter_file(void) {
    struct sock_filter instructions[256];
    size_t length = gate_build_filter(instructions, 256, 0);
    if (!length) die("construct seccomp filter");
    int fd = (int)syscall(SYS_memfd_create, "gate-seccomp", MFD_CLOEXEC | MFD_ALLOW_SEALING);
    if (fd < 0) die("create seccomp descriptor");
    size_t bytes = length * sizeof(instructions[0]);
    size_t done = 0;
    while (done < bytes) {
        ssize_t written = write(fd, (const char *)instructions + done, bytes - done);
        if (written < 0 && errno == EINTR) continue;
        if (written <= 0) die("write seccomp descriptor");
        done += (size_t)written;
    }
    if (lseek(fd, 0, SEEK_SET) < 0) die("rewind seccomp descriptor");
    if (fcntl(fd, F_ADD_SEALS, F_SEAL_SEAL | F_SEAL_SHRINK | F_SEAL_GROW | F_SEAL_WRITE)) die("seal seccomp descriptor");
    return fd;
}
static void must_be_regular(int dir, const char *name) {
    int fd = openat(dir, name, O_RDONLY | O_NOFOLLOW | O_CLOEXEC);
    if (fd < 0) die("open trusted runner file");
    struct stat st;
    if (fstat(fd, &st)) die("stat trusted runner file");
    close(fd);
    if (!S_ISREG(st.st_mode) || (st.st_mode & 0022)) reject("trusted runner files must be regular and not group/world writable");
}
int main(int argc, char **argv) {
    pid_t controller = getppid();
    if (prctl(PR_SET_PDEATHSIG, SIGKILL)) die("bind runner lifetime to its controller");
    if (getppid() != controller) reject("controller exited during runner startup");
    const char *job_arg = NULL, *output_arg = NULL;
    unsigned long timeout = 30, memory = 512, file_size = 16, cpu = 30;
    unsigned long expected_controller = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--help")) {
            puts("gate-sandbox --controller-pid PID --job ABS_DIR --output ABS_DIR [--timeout SEC --memory-mib N --file-mib N --cpu-seconds SEC]");
            return 0;
        }
        if (i + 1 >= argc) reject("option requires a value");
        const char *key = argv[i++], *value = argv[i];
        if (!strcmp(key, "--job")) { if (job_arg) reject("duplicate job"); job_arg = value; }
        else if (!strcmp(key, "--output")) { if (output_arg) reject("duplicate output"); output_arg = value; }
        else if (!strcmp(key, "--timeout")) timeout = number(value, 1, 3600);
        else if (!strcmp(key, "--memory-mib")) memory = number(value, 64, 8192);
        else if (!strcmp(key, "--file-mib")) file_size = number(value, 1, 1024);
        else if (!strcmp(key, "--cpu-seconds")) cpu = number(value, 1, 3600);
        else if (!strcmp(key, "--controller-pid")) { if (expected_controller) reject("duplicate controller PID"); expected_controller = number(value, 2, INT_MAX); }
        else reject("unknown option");
    }
    if (!job_arg || !output_arg) reject("both --job and --output are required");
    if (!expected_controller || controller != (pid_t)expected_controller || getppid() != (pid_t)expected_controller) reject("the expected controller is not the runner parent");
    if (getuid() == 0 || geteuid() != getuid() || getegid() != getgid()) reject("run as an ordinary unprivileged user");
    if (prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0)) die("set no-new-privileges");
    char job[PATH_MAX], output[PATH_MAX], native[PATH_MAX];
    int job_fd = directory(job_arg, job);
    int output_fd = directory(output_arg, output);
    if (is_within(job, output) || is_within(output, job)) reject("job and output directories must be disjoint");
    ssize_t executable_length = readlink("/proc/self/exe", native, sizeof(native) - 1);
    if (executable_length < 0 || executable_length >= (ssize_t)sizeof(native) - 1) die("resolve native launcher location");
    native[executable_length] = '\0';
    char *slash = strrchr(native, '/');
    if (!slash || slash == native) reject("invalid launcher directory");
    *slash = '\0';
    if (is_within(native, output) || is_within(output, native) || is_within(native, job) || is_within(job, native)) reject("trusted runner, job, and output directories must be disjoint");
    int native_fd = open(native, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
    if (native_fd < 0) die("open native launcher directory");
    must_be_regular(native_fd, "bootstrap.py");
    must_be_regular(native_fd, "libgate_seccomp.so");
    int seccomp_fd = filter_file();
    char job_num[24], output_num[24], native_num[24], seccomp_num[24];
    snprintf(job_num, sizeof(job_num), "%d", job_fd);
    snprintf(output_num, sizeof(output_num), "%d", output_fd);
    snprintf(native_num, sizeof(native_num), "%d", native_fd);
    snprintf(seccomp_num, sizeof(seccomp_num), "%d", seccomp_fd);
    char *sandbox_argv[] = {
        "/usr/bin/bwrap", "--unshare-user", "--unshare-pid", "--unshare-ipc", "--unshare-uts",
        "--disable-userns", "--assert-userns-disabled", "--die-with-parent", "--new-session",
        "--cap-drop", "ALL", "--clearenv",
        "--ro-bind", "/usr/lib", "/usr/lib", "--ro-bind", "/usr/bin/python3", "/usr/bin/python3",
        "--ro-bind", "/lib", "/lib", "--ro-bind", "/lib64", "/lib64",
        "--symlink", "usr/bin", "/bin",
        "--ro-bind-fd", job_num, "/job", "--bind-fd", output_num, "/output",
        "--ro-bind-fd", native_num, "/gate",
        "--proc", "/proc", "--remount-ro", "/proc", "--dev", "/dev",
        "--size", "67108864", "--tmpfs", "/dev/shm", "--size", "67108864", "--tmpfs", "/tmp",
        "--remount-ro", "/", "--chdir", "/output",
        "--setenv", "PATH", "/usr/bin:/bin", "--setenv", "LANG", "C.UTF-8",
        "--setenv", "LC_ALL", "C.UTF-8", "--setenv", "TZ", "UTC",
        "--setenv", "HOME", "/nonexistent", "--setenv", "GATE_JOB_DIR", "/job",
        "--setenv", "GATE_OUTPUT_DIR", "/output", "--setenv", "GATE_INSTRUCTION", "/job/instruction.bin",
        "--setenv", "GATE_ENVELOPE", "/job/envelope.json", "--setenv", "GATE_MANIFEST", "/job/manifest.json",
        "--seccomp", seccomp_num,
        "--", "/usr/bin/python3", "-I", "-S", "/gate/bootstrap.py", NULL
    };
    struct sigaction action = {0};
    action.sa_handler = on_signal;
    sigemptyset(&action.sa_mask);
    if (sigaction(SIGTERM, &action, NULL) || sigaction(SIGINT, &action, NULL) || sigaction(SIGHUP, &action, NULL)) die("install supervisor signal handler");
    pid_t supervisor = getpid();
    double started = monotonic_seconds();
    pid_t child = fork();
    if (child < 0) die("start sandbox");
    if (child == 0) {
        if (setpgid(0, 0)) die("create sandbox process group");
        if (prctl(PR_SET_PDEATHSIG, SIGKILL)) die("set parent-death signal");
        if (getppid() != supervisor) _exit(125);
        if (syscall(SYS_close_range, 3U, ~0U, CLOSE_RANGE_CLOEXEC)) die("close inherited descriptors on exec");
        int kept[] = { job_fd, output_fd, native_fd, seccomp_fd };
        for (size_t i = 0; i < sizeof(kept) / sizeof(kept[0]); i++) {
            if (fcntl(kept[i], F_SETFD, 0)) die("pass only required sandbox descriptors");
        }
        int null_fd = open("/dev/null", O_RDONLY | O_CLOEXEC);
        if (null_fd < 0 || dup2(null_fd, STDIN_FILENO) < 0) die("disconnect sandbox stdin");
        if (null_fd != STDIN_FILENO) close(null_fd);
        umask(0077);
        limit(RLIMIT_CORE, 0);
        limit(RLIMIT_AS, (rlim_t)memory * 1024U * 1024U);
        limit(RLIMIT_CPU, (rlim_t)cpu);
        limit(RLIMIT_FSIZE, (rlim_t)file_size * 1024U * 1024U);
        limit(RLIMIT_NOFILE, 64);
        limit(RLIMIT_NPROC, 64);
        char *empty_environment[] = { NULL };
        execve(sandbox_argv[0], sandbox_argv, empty_environment);
        die("execute required bubblewrap runtime");
    }
    close(job_fd); close(output_fd); close(native_fd); close(seccomp_fd);
    int status = 0, timed_out = 0;
    for (;;) {
        pid_t completed = waitpid(child, &status, WNOHANG);
        if (completed == child) break;
        if (completed < 0 && errno != EINTR) die("wait for sandbox");
        if (interrupted || monotonic_seconds() - started >= (double)timeout) {
            timed_out = !interrupted;
            if (kill(-child, SIGKILL) && errno != ESRCH) die("terminate sandbox process group");
            if (kill(child, SIGKILL) && errno != ESRCH) die("terminate sandbox supervisor");
            while (waitpid(child, &status, 0) < 0) {
                if (errno != EINTR) die("reap terminated sandbox");
            }
            break;
        }
        struct timespec interval = { .tv_sec = 0, .tv_nsec = 20000000L };
        while (nanosleep(&interval, &interval) && errno == EINTR && !interrupted) {}
    }
    if (timed_out) { fputs("gate-sandbox: wall-time limit exceeded\n", stderr); return 124; }
    if (interrupted) return 128 + interrupted;
    if (WIFEXITED(status)) return WEXITSTATUS(status);
    if (WIFSIGNALED(status)) return 128 + WTERMSIG(status);
    reject("sandbox returned an unexpected process status");
}
