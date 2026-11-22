#define _GNU_SOURCE
#include "seccomp_policy.h"
#include <errno.h>
#include <linux/audit.h>
#include <linux/close_range.h>
#include <linux/seccomp.h>
#include <stddef.h>
#include <stdint.h>
#include <sys/prctl.h>
#include <sys/syscall.h>
#include <unistd.h>

#if defined(__x86_64__)
#define GATE_ARCH AUDIT_ARCH_X86_64
#elif defined(__aarch64__)
#define GATE_ARCH AUDIT_ARCH_AARCH64
#else
#error "The syscall architecture must be reviewed before enabling this runner"
#endif

static int append(struct sock_filter *out, size_t cap, size_t *n,
                  struct sock_filter instruction) {
    if (*n >= cap) { errno = E2BIG; return -1; }
    out[(*n)++] = instruction;
    return 0;
}

#define EMIT(x) do { if (append(out, capacity, &n, (struct sock_filter)x)) return 0; } while (0)
#define DENY_NR(x) do { \
    EMIT(BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, (x), 0, 1)); \
    EMIT(BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ERRNO | EPERM)); \
} while (0)

size_t gate_build_filter(struct sock_filter *out, size_t capacity, int worker) {
    size_t n = 0;
    EMIT(BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(struct seccomp_data, arch)));
    EMIT(BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, GATE_ARCH, 1, 0));
    EMIT(BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_KILL_PROCESS));
    EMIT(BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(struct seccomp_data, nr)));
#if defined(__x86_64__)
    EMIT(BPF_JUMP(BPF_JMP | BPF_JGE | BPF_K, 0x40000000U, 0, 1));
    EMIT(BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_KILL_PROCESS));
#endif
#ifdef SYS_socket
    DENY_NR(SYS_socket);
#endif
#ifdef SYS_socketpair
    DENY_NR(SYS_socketpair);
#endif
#ifdef SYS_connect
    DENY_NR(SYS_connect);
#endif
#ifdef SYS_accept
    DENY_NR(SYS_accept);
#endif
#ifdef SYS_accept4
    DENY_NR(SYS_accept4);
#endif
#ifdef SYS_bind
    DENY_NR(SYS_bind);
#endif
#ifdef SYS_listen
    DENY_NR(SYS_listen);
#endif
#ifdef SYS_sendto
    DENY_NR(SYS_sendto);
#endif
#ifdef SYS_recvfrom
    DENY_NR(SYS_recvfrom);
#endif
#ifdef SYS_sendmsg
    DENY_NR(SYS_sendmsg);
#endif
#ifdef SYS_recvmsg
    DENY_NR(SYS_recvmsg);
#endif
#ifdef SYS_sendmmsg
    DENY_NR(SYS_sendmmsg);
#endif
#ifdef SYS_recvmmsg
    DENY_NR(SYS_recvmmsg);
#endif
#ifdef SYS_shutdown
    DENY_NR(SYS_shutdown);
#endif
#ifdef SYS_getsockname
    DENY_NR(SYS_getsockname);
#endif
#ifdef SYS_getpeername
    DENY_NR(SYS_getpeername);
#endif
#ifdef SYS_setsockopt
    DENY_NR(SYS_setsockopt);
#endif
#ifdef SYS_getsockopt
    DENY_NR(SYS_getsockopt);
#endif
#ifdef SYS_io_uring_setup
    DENY_NR(SYS_io_uring_setup);
    DENY_NR(SYS_io_uring_enter);
    DENY_NR(SYS_io_uring_register);
#endif
#ifdef SYS_ptrace
    DENY_NR(SYS_ptrace);
#endif
#ifdef SYS_process_vm_readv
    DENY_NR(SYS_process_vm_readv);
    DENY_NR(SYS_process_vm_writev);
#endif
#ifdef SYS_pidfd_getfd
    DENY_NR(SYS_pidfd_getfd);
#endif
#ifdef SYS_open_by_handle_at
    DENY_NR(SYS_open_by_handle_at);
#endif
#ifdef SYS_name_to_handle_at
    DENY_NR(SYS_name_to_handle_at);
#endif
#ifdef SYS_mount
    DENY_NR(SYS_mount);
#endif
#ifdef SYS_umount2
    DENY_NR(SYS_umount2);
#endif
#ifdef SYS_pivot_root
    DENY_NR(SYS_pivot_root);
#endif
#ifdef SYS_chroot
    DENY_NR(SYS_chroot);
#endif
#ifdef SYS_unshare
    DENY_NR(SYS_unshare);
#endif
#ifdef SYS_setns
    DENY_NR(SYS_setns);
#endif
#ifdef SYS_open_tree
    DENY_NR(SYS_open_tree);
#endif
#ifdef SYS_move_mount
    DENY_NR(SYS_move_mount);
#endif
#ifdef SYS_fsopen
    DENY_NR(SYS_fsopen);
    DENY_NR(SYS_fsconfig);
    DENY_NR(SYS_fsmount);
    DENY_NR(SYS_fspick);
#endif
#ifdef SYS_mount_setattr
    DENY_NR(SYS_mount_setattr);
#endif
#ifdef SYS_bpf
    DENY_NR(SYS_bpf);
#endif
#ifdef SYS_perf_event_open
    DENY_NR(SYS_perf_event_open);
#endif
#ifdef SYS_userfaultfd
    DENY_NR(SYS_userfaultfd);
#endif
#ifdef SYS_add_key
    DENY_NR(SYS_add_key);
    DENY_NR(SYS_request_key);
    DENY_NR(SYS_keyctl);
#endif
#ifdef SYS_reboot
    DENY_NR(SYS_reboot);
#endif
#ifdef SYS_kexec_load
    DENY_NR(SYS_kexec_load);
#endif
#ifdef SYS_kexec_file_load
    DENY_NR(SYS_kexec_file_load);
#endif
#ifdef SYS_init_module
    DENY_NR(SYS_init_module);
#endif
#ifdef SYS_finit_module
    DENY_NR(SYS_finit_module);
#endif
#ifdef SYS_delete_module
    DENY_NR(SYS_delete_module);
#endif
#ifdef SYS_swapon
    DENY_NR(SYS_swapon);
    DENY_NR(SYS_swapoff);
#endif
#ifdef SYS_iopl
    DENY_NR(SYS_iopl);
#endif
#ifdef SYS_ioperm
    DENY_NR(SYS_ioperm);
#endif
#ifdef SYS_mknod
    DENY_NR(SYS_mknod);
#endif
#ifdef SYS_mknodat
    DENY_NR(SYS_mknodat);
#endif
#ifdef SYS_fanotify_init
    DENY_NR(SYS_fanotify_init);
#endif
    if (worker) {
#ifdef SYS_execve
        DENY_NR(SYS_execve);
#endif
#ifdef SYS_execveat
        DENY_NR(SYS_execveat);
#endif
#ifdef SYS_fork
        DENY_NR(SYS_fork);
#endif
#ifdef SYS_vfork
        DENY_NR(SYS_vfork);
#endif
#ifdef SYS_clone
        DENY_NR(SYS_clone);
#endif
#ifdef SYS_clone3
        DENY_NR(SYS_clone3);
#endif
    }
    EMIT(BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW));
    return n;
}

int gate_worker_policy_version(void) { return 1; }

int gate_install_worker_filter(void) {
    struct sock_filter instructions[256];
    size_t length = gate_build_filter(instructions, 256, 1);
    if (!length || length > UINT16_MAX) { errno = EINVAL; return -1; }
    if (syscall(SYS_close_range, 3U, ~0U, 0U)) return -1;
    if (prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0)) return -1;
    struct sock_fprog program = { .len = (unsigned short)length, .filter = instructions };
    return (int)syscall(SYS_seccomp, SECCOMP_SET_MODE_FILTER, SECCOMP_FILTER_FLAG_TSYNC, &program);
}
