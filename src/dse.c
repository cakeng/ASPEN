#include "dse.h"

static _Atomic unsigned int dse_thread_id_counter = 0;

void *dse_thread_runtime (void* thread_info)
{
    dse_t *dse = (dse_t*) thread_info;
    pthread_mutex_lock(&dse->thread_mutex);
    atomic_store (&dse->run, 0);
    pthread_cond_wait(&dse->thread_cond, &dse->thread_mutex); 
    while (dse->kill == 0 || dse->target != NULL)
    {
        dse_schedule (dse);
        if (dse->run == 0 && dse->target == NULL)
            pthread_cond_wait(&dse->thread_cond, &dse->thread_mutex); 
    }
    return NULL;
}

int dse_check_compute (dse_t *dse, ninst_t *ninst)
{
    #ifdef DEBUG
    if (dse == NULL || ninst == NULL)
    {
        ERROR_PRTF ("Error: Invalid arguments to dse_check_compute()");
        assert (0);
    }
    if (ninst->state != NINST_READY)
    {
        ERROR_PRTF ("Error: ninst->state != NINST_READY in dse_check_compute() - State: %s", 
            ninst_state_str[ninst->state]);
        assert (0);
    }
    #endif
    if (ninst->ldata->nasm->num_peers < 2)
        return 1;
    HASH_t group_hash = dse->dse_group->my_peer_data->peer_hash;
    if (check_ninst_compute_using_peer_idx (ninst, get_peer_idx(ninst->ldata->nasm, group_hash)))
        return 1;
    HASH_t dse_hash = dse->my_peer_data->peer_hash;
    if (check_ninst_compute_using_peer_idx (ninst, get_peer_idx(ninst->ldata->nasm, dse_hash)))
        return 1;
    return 0;
}

void dse_push_to_tx (dse_t *dse, ninst_t *ninst)
{
    #ifdef DEBUG
    if (dse == NULL || ninst == NULL)
    {
        ERROR_PRTF ("Error: Invalid arguments to dse_check_send()");
        assert (0);
    }
    if (ninst->state != NINST_COMPLETED)
    {
        ERROR_PRTF ("Error: ninst->state != NINST_COMPLETED in dse_check_send() - State: %s", 
            ninst_state_str[ninst->state]);
        assert (0);
    }
    #endif
    if (ninst->ldata->nasm->num_peers < 2)
        return;
    HASH_t group_hash = dse->dse_group->my_peer_data->peer_hash;
    HASH_t dse_hash = dse->my_peer_data->peer_hash;
    for (int i = 0; i < ninst->ldata->nasm->num_peers; i++)
    {
        if (i == group_hash || i == dse_hash)
            continue;
        if (check_ninst_send_using_peer_idx (ninst, i))
        {
            aspen_peer_t *peer = ninst->ldata->nasm->peer_map[i];
            pthread_mutex_lock (&peer->tx_queue->occupied_mutex);
            push_ninsts_to_queue (peer->tx_queue, &ninst, 1);
            pthread_mutex_unlock (&peer->tx_queue->occupied_mutex);
        }
    }
}

void dse_execute (dse_t *dse, ninst_t *ninst)
{
    switch (ninst->ldata->layer->type)
    {
        case CONV_LAYER:
            tiled_conv2d (ninst, dse);
            break;
        case MAXPOOL_LAYER:
            tiled_maxpool2d (ninst, dse);
            break;
        case AVGPOOL_LAYER:
            tiled_avgpool2d (ninst, dse);
            break;
        case FC_LAYER:
            tiled_fully_connected (ninst, dse);
            break;
        case RESIDUAL_LAYER:
            tiled_residual (ninst, dse);
            break;
        case SOFTMAX_LAYER:
            tiled_softmax (ninst, dse);
            break;
        case YOLO_LAYER:
            tiled_yolo (ninst, dse);
            break;
        case APPEND_LAYER:
            tiled_append (ninst, dse);
            break;
        case MATMUL_LAYER:
            tiled_matmul (ninst, dse);
            break;
        case LAYERNORM_LAYER:
            tiled_layernorm (ninst, dse);
            break;
        case K_ATTENTION_LAYER:
            tiled_k_attention (ninst, dse);
            break;
        case V_ATTENTION_LAYER:
            tiled_v_attention (ninst, dse);
            break;
        default:
            break;
    }
}

void dse_schedule (dse_t *dse)
{
    if (dse->target == NULL && (dse->run != 0 && dse->kill == 0))
    {
        rpool_fetch_ninsts (dse->rpool, &dse->target, 1, 0);
    }
    if (dse->target == NULL)
        return;

    // Execute.
    ninst_t *ninst = dse->target;
    dse->target = NULL;
    #ifdef DEBUG 
    if (ninst->state != NINST_READY)
    {
        ERROR_PRTF ("Error: ninst->state != NINST_READY in dse_thread_runtime()");
        assert (0);
    }
    #endif
    if (!dse_check_compute(dse, ninst))
        return;
    dse_execute (dse, ninst);
    ninst->state = NINST_COMPLETED;
    unsigned int num_ninst_completed = atomic_fetch_add (&ninst->ldata->num_ninst_completed, 1);
    if (num_ninst_completed == ninst->ldata->num_ninst - 1)
    {
        for (int pidx = 0; pidx < NUM_PARENT_ELEMENTS; pidx++)
        {
            if (ninst->ldata->parent_ldata_idx_arr[pidx] == -1)
                continue;
            nasm_ldata_t *parent_ldata = &ninst->ldata->nasm->ldata_arr[ninst->ldata->parent_ldata_idx_arr[pidx]];
            unsigned int num_child_ldata_completed = atomic_fetch_add (&parent_ldata->num_child_ldata_completed, 1);
            if (num_child_ldata_completed + 1 == parent_ldata->num_child_ldata && (parent_ldata != parent_ldata->nasm->ldata_arr))
                free_ldata_out_mat (parent_ldata);
        }
        
        nasm_t *nasm = ninst->ldata->nasm;
        unsigned int num_ldata_completed = atomic_fetch_add (&nasm->num_ldata_completed, 1);
        if (num_ldata_completed == nasm->num_ldata - 1)
        {
            // All layers of the nasm is completed.
            atomic_store (&nasm->completed, 1);
            rpool_queue_group_t *rpool_queue_group 
                = get_queue_group_from_nasm (dse->rpool, ninst->ldata->nasm);
            pthread_mutex_lock (&nasm->nasm_mutex);
            pthread_cond_signal (&nasm->nasm_cond);
            pthread_mutex_unlock (&nasm->nasm_mutex);
        }
    }
    update_children_but_prioritize_dse_target (dse->rpool, ninst, dse);
    dse_push_to_tx (dse, ninst);
}

runtime_profile_t *dse_profile_init (HASH_t ninst_hash, size_t runtime_usec)
{
    runtime_profile_t *profile = (runtime_profile_t *) calloc (1, sizeof (runtime_profile_t));
    profile->ninst_hash = ninst_hash;
    profile->runtime_usec = runtime_usec;
    return profile;
}

void dse_profile_destroy (runtime_profile_t *dse_profile)
{
    if (dse_profile == NULL)
        return;
    free (dse_profile);
}

void dse_group_add_profile (dse_group_t *dse_group, ninst_t *ninst)
{
    if (dse_group == NULL)
    {
        ERROR_PRTF ("ERROR: dse_group_add_profile: dse_group is NULL");
        assert (0);
    }
    HASH_t ninst_hash = ninst->type_hash;
    ssize_t runtime_usec = dse_group_get_profile (dse_group, ninst);
    if (runtime_usec != -1)
        return;

    // Allocate memory
    int free_flag = 0;
    if (ninst->ldata->out_mat == NULL)
    {
        free_flag = 1;
        alloc_ldata_out_mat (ninst->ldata);
    }
    int *free_flag_parent= calloc (ninst->num_parent_ninsts, sizeof (int));
    for (int i = 0; i < ninst->num_parent_ninsts; i++)
    {
        ninst_t *parent_ninst = ninst->ldata->nasm->ninst_arr + ninst->parent_ninst_idx_arr[i];
        if (parent_ninst->ldata->out_mat == NULL)
        {
            free_flag_parent[i] = 1;
            alloc_ldata_out_mat (parent_ninst->ldata);
        }
    }
    
    // Execute
    size_t start = get_elapsed_usec ();
    for (int i = 0; i < dse_group->num_dses; i++)
        for (int j = 0; j < DSE_PROFILE_RUN_NUM; j++)
            dse_execute (&dse_group->dse_arr[i], ninst);
    runtime_usec = get_elapsed_usec () - start;
    runtime_usec /= DSE_PROFILE_RUN_NUM;
    runtime_usec /= dse_group->num_dses;
    runtime_usec /= dse_group->num_dses;
    runtime_usec = runtime_usec < 1 ? 1 : runtime_usec;

    // Add profile
    runtime_profile_t *profile = dse_profile_init (ninst_hash, runtime_usec);
    dse_group->profile_arr[dse_group->num_profiles++] = profile;
    if (dse_group->num_profiles == dse_group->max_num_profiles)
    {
        dse_group->max_num_profiles *= 2;
        dse_group->profile_arr = (runtime_profile_t **) realloc (dse_group->profile_arr, dse_group->max_num_profiles * sizeof (runtime_profile_t *));
    }

    // Free memory
    if (free_flag)
        free_ldata_out_mat (ninst->ldata);
    for (int i = 0; i < ninst->num_parent_ninsts; i++)
    {
        ninst_t *parent_ninst = ninst->ldata->nasm->ninst_arr + ninst->parent_ninst_idx_arr[i];
        if (free_flag_parent[i])
            free_ldata_out_mat (parent_ninst->ldata);
    }
    printf ("Profiled Ninst %08lx: Runtime %lu usecs.\n", ninst_hash, runtime_usec);
    print_ninst_info (ninst, 0);

}

ssize_t dse_group_get_profile (dse_group_t *dse_group, ninst_t *ninst)
{
    if (dse_group == NULL)
    {
        ERROR_PRTF ("ERROR: dse_group_get_ninst_time: dse_group is NULL");
        assert (0);
    }
    HASH_t ninst_hash = ninst->type_hash;
    for (int i = 0; i < dse_group->num_profiles; i++)
    {
        if (dse_group->profile_arr[i]->ninst_hash == ninst_hash)
            return dse_group->profile_arr[i]->runtime_usec;
    }
    return -1;
}

void dse_group_profile_nasm (dse_group_t *dse_group, nasm_t *nasm)
{
    if (dse_group == NULL)
    {
        ERROR_PRTF ("ERROR: dse_group_profile_nasm: dse_group is NULL");
        assert (0);
    }
    if (nasm == NULL)
    {
        ERROR_PRTF ("ERROR: dse_group_profile_nasm: nasm is NULL");
        assert (0);
    }
    for (int i = 0; i < nasm->num_ninst; i++)
    {
        ninst_t *ninst = &nasm->ninst_arr[i];
        dse_group_add_profile (dse_group, ninst);
    }
}

void dse_group_load_profile_data (dse_group_t *dse_group, char *filename)
{
    if (dse_group == NULL)
    {
        ERROR_PRTF ("ERROR: dse_group_load_profile_data: dse_group is NULL");
        assert (0);
    }
    if (filename == NULL)
    {
        ERROR_PRTF ("ERROR: dse_group_load_profile_data: filename is NULL");
        assert (0);
    }
    FILE *fp = fopen (filename, "r");
    if (fp == NULL)
    {
        ERRNO_PRTF ("ERROR: dse_group_load_profile_data: Unable to open file %s", filename);
        assert (0);
    }
    while (!feof (fp))
    {
        HASH_t ninst_hash;
        size_t runtime_usec;
        int ret = fscanf (fp, "%08lx, %lu\n", &ninst_hash, &runtime_usec);
        if (ret != 2)
        {
            ERRNO_PRTF ("ERROR: dse_group_load_profile_data: fscanf failed");
            assert (0);
        }
        runtime_profile_t *profile = dse_profile_init (ninst_hash, runtime_usec);
        dse_group->profile_arr[dse_group->num_profiles++] = profile;
        if (dse_group->num_profiles == dse_group->max_num_profiles)
        {
            dse_group->max_num_profiles *= 2;
            dse_group->profile_arr = (runtime_profile_t **) realloc (dse_group->profile_arr, dse_group->max_num_profiles * sizeof (runtime_profile_t *));
        }
    }
    fclose (fp);
}

void dse_group_save_profile_data (dse_group_t *dse_group, char *filename)
{
    if (dse_group == NULL)
    {
        ERROR_PRTF ("ERROR: dse_group_save_profile_data: dse_group is NULL");
        assert (0);
    }
    if (filename == NULL)
    {
        ERROR_PRTF ("ERROR: dse_group_load_profile_data: filename is NULL");
        assert (0);
    }
    FILE *fp = fopen (filename, "w");
    if (fp == NULL)
    {
        ERRNO_PRTF ("ERROR: dse_group_save_profile_data: Unable to open file %s", filename);
        assert (0);
    }
    for (int i = 0; i < dse_group->num_profiles; i++)
    {
        runtime_profile_t *profile = dse_group->profile_arr[i];
        fprintf (fp, "%08lx, %lu\n", profile->ninst_hash, profile->runtime_usec);
    }
    fclose (fp);
}

void dse_group_nasm_export_heft_data (dse_group_t *dse_group, nasm_t *nasm, char *filename)
{
    if (dse_group == NULL)
    {
        ERROR_PRTF ("ERROR: dse_group_save_profile_data: dse_group is NULL");
        assert (0);
    }
    if (filename == NULL)
    {
        ERROR_PRTF ("ERROR: dse_group_load_profile_data: filename is NULL");
        assert (0);
    }
    FILE *fp = fopen (filename, "w");
    if (fp == NULL)
    {
        ERRNO_PRTF ("ERROR: dse_group_save_profile_data: Unable to open file %s", filename);
        assert (0);
    }
    fprintf (fp, "%d\n", nasm->num_ninst);
    fprintf (fp, "2\n");
    for (int i = 0; i < nasm->num_ninst; i++)
        fprintf (fp, "%ld ", dse_group_get_profile (dse_group, &nasm->ninst_arr[i]));
    fprintf (fp, "\n");
    for (int i = 0; i < nasm->num_ninst; i++)
        fprintf (fp, "%ld ", dse_group_get_profile (dse_group, &nasm->ninst_arr[i]));
    fprintf (fp, "\n");
    for (int i = 0; i < nasm->num_ninst; i++)
    {
        ninst_t *ninst = &nasm->ninst_arr[i];
        for (int j = 0; j < nasm->num_ninst; j++)
        {
            int printed = 0;
            ninst_t *child = &nasm->ninst_arr[j];
            for (int k = 0; k < ninst->num_child_ninsts; k++)
            {
                if (child == ninst->child_ninst_arr[k])
                {
                    printed = 0;
                    fprintf (fp, "%ld ", ninst->tile_dims[OUT_H] * ninst->tile_dims[OUT_W] * sizeof(float) 
                        /12  + 800);
                    break;
                }
            }
            if (printed == 0)
                fprintf (fp, "0 ");
        }
        fprintf (fp, "\n");
    }
    fclose (fp);
}

dse_group_t *dse_group_init (unsigned int num_dse)
{
    dse_group_t *dse_group = (dse_group_t *) calloc (1, sizeof (dse_group_t));
    dse_group->my_peer_data = peer_init ();
    dse_group->num_dses = num_dse;
    dse_group->dse_arr = (dse_t *) calloc (num_dse, sizeof (dse_t));
    for (int i = 0; i < num_dse; i++)
    {
        dse_init (&dse_group->dse_arr[i]);
        dse_group->dse_arr[i].dse_group = dse_group;
    }
    dse_group->max_num_profiles = 32;
    dse_group->num_profiles = 0;
    dse_group->profile_arr = (runtime_profile_t **) calloc (dse_group->max_num_profiles, sizeof (runtime_profile_t *));
    return dse_group;
}

void dse_group_set_rpool (dse_group_t *dse_group, rpool_t *rpool)
{
    if (dse_group == NULL)
    {
        ERROR_PRTF ("ERROR: dse_group_set_rpool: dse_group is NULL");
        assert (0);
    }
    if (rpool == NULL)
    {
        ERROR_PRTF ("ERROR: dse_group_set_rpool: rpool is NULL");
        assert (0);
    }
    for (int i = 0; i < dse_group->num_dses; i++)
    {
        dse_group->dse_arr[i].rpool = rpool;
    }
    add_ref_dses (rpool, dse_group->num_dses);
}

void dse_group_destroy (dse_group_t *dse_group)
{
    if (dse_group == NULL)
        return;
    for (int i = 0; i < dse_group->num_dses; i++)
    {
        if (dse_group->dse_arr[i].rpool != NULL)
            atomic_fetch_sub (&dse_group->dse_arr[i].rpool->ref_dses, 1);
        dse_destroy (&dse_group->dse_arr[i]);
    }
    free (dse_group->dse_arr);
    destroy_peer (dse_group->my_peer_data);
    for (int i = 0 ; i < dse_group->max_num_profiles; i++)
        dse_profile_destroy (dse_group->profile_arr[i]);
    free (dse_group->profile_arr);
    free (dse_group);
}

void dse_init (dse_t *dse)
{
    if (dse == NULL)
    {
        ERROR_PRTF ("ERROR: dse_init: dse is NULL");
        assert (0);
    }
    dse->dse_group = NULL;
    dse->thread_id = atomic_fetch_add (&dse_thread_id_counter, 1);
    dse->rpool = NULL;
    dse->scratchpad = aspen_calloc (DSE_SCRATCHPAD_SIZE, 1);
    if (dse->scratchpad == NULL)
    {
        ERRNO_PRTF ("ERROR: dse_init: Unable to allocate scratchpad");
        assert (0);
    }
    dse->thread_mutex = (pthread_mutex_t)PTHREAD_MUTEX_INITIALIZER;
    dse->thread_cond = (pthread_cond_t)PTHREAD_COND_INITIALIZER;
    dse->ninst_cache = calloc (1, sizeof (rpool_queue_t));
    rpool_init_queue (dse->ninst_cache);
    atomic_store (&dse->run, -1);
    atomic_store (&dse->kill, 0);
    dse->my_peer_data = peer_init ();
    pthread_create (&dse->thread, NULL, dse_thread_runtime, (void*)dse);
}

void dse_destroy (dse_t *dse)
{
    if (dse == NULL)
        return;
    if (atomic_load (&dse->run) == 1)
    {
        ERROR_PRTF ("ERROR: Tried to destroy dse while it is running.");
    }
    atomic_store (&dse->kill, 1);
    if (atomic_load (&dse->run) != 1)
        dse_run (dse);
    pthread_join (dse->thread, NULL);
    pthread_mutex_destroy (&dse->thread_mutex);
    pthread_cond_destroy (&dse->thread_cond);
    if (dse->scratchpad != NULL)
        aspen_free (dse->scratchpad);
    rpool_destroy_queue (dse->ninst_cache);
    destroy_peer (dse->my_peer_data);
    free (dse->ninst_cache);
}

void dse_run (dse_t *dse)
{
    if (dse == NULL)
    {
        ERROR_PRTF ("ERROR: dse_run: dse is NULL");
        return;
    }
    int state = atomic_load (&dse->run);
    while (state == -1)
        state = atomic_load (&dse->run);
    if (state == 1)
    {
        return;
    }
    else 
    {
        pthread_mutex_lock (&dse->thread_mutex);
        atomic_store (&dse->run, 1);
        pthread_cond_signal (&dse->thread_cond);
        pthread_mutex_unlock (&dse->thread_mutex);
    }
}

void dse_stop (dse_t *dse)
{
    if (dse == NULL)
    {
        ERROR_PRTF ("ERROR: dse_stop: dse is NULL");
        return;
    }
    unsigned int state = atomic_exchange (&dse->run, 0);
    if (state == 0)
    {
        return;
    }
    else 
    {
        pthread_mutex_lock (&dse->thread_mutex);
        pthread_mutex_unlock (&dse->thread_mutex);
    }
}

void dse_group_run (dse_group_t *dse_group)
{
    if (dse_group == NULL)
    {
        ERROR_PRTF ("ERROR: dse_group_run: dse_group is NULL");
        assert (0);
    }
    for (int i = 0; i < dse_group->num_dses; i++)
    {
        dse_run (&dse_group->dse_arr[i]);
    }
}

void dse_group_stop (dse_group_t *dse_group)
{
    if (dse_group == NULL)
    {
        ERROR_PRTF ("ERROR: dse_group_stop: dse_group is NULL");
        assert (0);
    }
    for (int i = 0; i < dse_group->num_dses; i++)
    {
        dse_stop (&dse_group->dse_arr[i]);
    }
}

unsigned int dse_check_nasm_completion (nasm_t *nasm)
{
    #ifdef DEBUG
    if (nasm == NULL)
    {
        ERROR_PRTF ("ERROR: dse_check_nasm_completion: nasm is NULL");
        assert (0);
    }
    #endif
    return atomic_load (&nasm->completed);
}

void dse_group_run_until_nasm_completion (dse_group_t *dse_group, nasm_t *nasm)
{
    dse_group_run (dse_group);
    dse_wait_for_nasm_completion (nasm);
    dse_group_stop (dse_group);
}

void dse_wait_for_nasm_completion (nasm_t *nasm)
{
    pthread_mutex_lock (&nasm->nasm_mutex);
    if (dse_check_nasm_completion (nasm) == 1)
    {
        pthread_mutex_unlock (&nasm->nasm_mutex);
        return;
    }
    pthread_cond_wait (&nasm->nasm_cond, &nasm->nasm_mutex);
    pthread_mutex_unlock (&nasm->nasm_mutex);
}

void update_children (rpool_t *rpool, ninst_t *ninst, unsigned int dse_idx)
{
    #ifdef DEBUG
    if (rpool == NULL || ninst == NULL)
    {
        ERROR_PRTF ("Error: Invalid arguments to dse_update_children()");
        assert (0);
    }
    if (ninst->state != NINST_COMPLETED)
    {
        ERROR_PRTF ("Error: ninst->state != NINST_STATE_COMPLETED in dse_update_children()");
        assert (0);
    }
    #endif
    for (int i = 0; i < ninst->num_child_ninsts; i++)
    {
        ninst_t *child_ninst = ninst->child_ninst_arr[i];
        unsigned int num_parent_ninsts_completed = atomic_fetch_add (&child_ninst->num_parent_ninsts_completed, 1);
        if (num_parent_ninsts_completed == child_ninst->num_parent_ninsts - 1)
        {
            #ifdef DEBUG
            if (child_ninst->state != NINST_NOT_READY)
            {
                ERROR_PRTF ("Error: child_ninst->state != NINST_NOT_READY in dse_update_children()");
                assert (0);
            }
            #endif
            child_ninst->state = NINST_READY;
            rpool_push_ninsts (rpool, &child_ninst, 1, 0);
            if (child_ninst->ldata->out_mat == NULL)
                alloc_ldata_out_mat (child_ninst->ldata);
        }
    }
}

void update_children_to_cache (rpool_queue_t *cache, ninst_t *ninst)
{
    #ifdef DEBUG
    if (cache == NULL || ninst == NULL)
    {
        ERROR_PRTF ("Error: Invalid arguments to dse_update_children_to_cache()");
        assert (0);
    }
    if (ninst->state != NINST_COMPLETED)
    {
        ERROR_PRTF ("Error: ninst->state != NINST_STATE_COMPLETED in dse_update_children_to_cache()");
        assert (0);
    }
    #endif
    if (ninst->state != NINST_COMPLETED)
        return;
    for (int i = 0; i < ninst->num_child_ninsts; i++)
    {
        ninst_t *child_ninst = ninst->child_ninst_arr[i];
        unsigned int num_parent_ninsts_completed = atomic_fetch_add (&child_ninst->num_parent_ninsts_completed, 1);
        if (num_parent_ninsts_completed == child_ninst->num_parent_ninsts - 1)
        {
            #ifdef DEBUG 
            if (child_ninst->state != NINST_NOT_READY)
            {
                ERROR_PRTF ("Error: child_ninst->state != NINST_NOT_READY in dse_update_children_to_cache()");
                assert (0);
            }
            #endif
            child_ninst->state = NINST_READY;
            push_ninsts_to_queue (cache, &child_ninst, 1);
            if (child_ninst->ldata->out_mat == NULL)
                alloc_ldata_out_mat (child_ninst->ldata);
        }
    }
}

void update_children_but_prioritize_dse_target (rpool_t *rpool, ninst_t *ninst, dse_t *dse)
{
    #ifdef DEBUG
    if (ninst->state != NINST_COMPLETED)
    {
        ERROR_PRTF ("Error: ninst->state != NINST_STATE_COMPLETED in dse_update_children_to_cache()");
        assert (0);
    }
    #endif
    if (ninst->state != NINST_COMPLETED)
        return;
    ninst_t **cache = dse->scratchpad;
    unsigned int num_cache = 0;
    for (int i = 0; i < ninst->num_child_ninsts; i++)
    {
        ninst_t *child_ninst = ninst->child_ninst_arr[i];
        unsigned int num_parent_ninsts_completed = atomic_fetch_add (&child_ninst->num_parent_ninsts_completed, 1);
        if (num_parent_ninsts_completed == child_ninst->num_parent_ninsts - 1)
        {
            #ifdef DEBUG 
            if (child_ninst->state != NINST_NOT_READY)
            {
                ERROR_PRTF ("Error: child_ninst->state != NINST_NOT_READY in dse_update_children_to_cache()");
                assert (0);
            }
            #endif
            child_ninst->state = NINST_READY;
            if (dse->target != NULL)
            {
                cache[num_cache++] = child_ninst;
            }
            else
                dse->target = child_ninst;
            if (child_ninst->ldata->out_mat == NULL)
                alloc_ldata_out_mat (child_ninst->ldata);
        }
    }
    rpool_push_ninsts (rpool, cache, num_cache, 0);
}

void update_children_to_cache_but_prioritize_dse_target (rpool_queue_t *cache, ninst_t *ninst, ninst_t **dse_target)
{
    #ifdef DEBUG
    if (cache == NULL || ninst == NULL)
    {
        ERROR_PRTF ("Error: Invalid arguments to dse_update_children_to_cache()");
        assert (0);
    }
    if (ninst->state != NINST_COMPLETED)
    {
        ERROR_PRTF ("Error: ninst->state != NINST_STATE_COMPLETED in dse_update_children_to_cache()");
        assert (0);
    }
    #endif
    if (ninst->state != NINST_COMPLETED)
        return;
    for (int i = 0; i < ninst->num_child_ninsts; i++)
    {
        ninst_t *child_ninst = ninst->child_ninst_arr[i];
        unsigned int num_parent_ninsts_completed = atomic_fetch_add (&child_ninst->num_parent_ninsts_completed, 1);
        if (num_parent_ninsts_completed == child_ninst->num_parent_ninsts - 1)
        {
            #ifdef DEBUG 
            if (child_ninst->state != NINST_NOT_READY)
            {
                ERROR_PRTF ("Error: child_ninst->state != NINST_NOT_READY in dse_update_children_to_cache()");
                assert (0);
            }
            #endif
            child_ninst->state = NINST_READY;
            if (*dse_target != NULL)
                push_ninsts_to_queue (cache, &child_ninst, 1);
            else
                *dse_target = child_ninst;
            if (child_ninst->ldata->out_mat == NULL)
                alloc_ldata_out_mat (child_ninst->ldata);
        }
    }
}

void push_first_layer_to_rpool (rpool_t *rpool, nasm_t *nasm, void* input_data)
{
    #ifdef DEBUG
    if (rpool == NULL || nasm == NULL)
    {
        ERROR_PRTF ("Error: Invalid arguments to dse_push_first_layer_to_rpool()");
        assert (0);
    }
    #endif
    nasm_ldata_t *ldata = &nasm->ldata_arr[0];
    alloc_ldata_out_mat (ldata);
    if (input_data != NULL)
        copy_buffer_to_ldata_out_mat (ldata, input_data);
    for (int i = 0; i < ldata->num_ninst; i++)
    {
        ninst_t *ninst = &ldata->ninst_arr_start[i];
        if (ninst->state != NINST_NOT_READY)
        {
            ERROR_PRTF ("Error: ninst->state != NINST_NOT_READY in dse_push_first_layer_to_rpool()");
            assert (0);
        }
        ninst->state = NINST_COMPLETED;
        atomic_fetch_add (&ninst->ldata->num_ninst_completed , 1);
        int num_dse = rpool->ref_dses > 0 ? rpool->ref_dses : 1;
        update_children (rpool, ninst, ldata->num_ninst > num_dse? i/(ldata->num_ninst/num_dse) : i);
    }
    atomic_fetch_add (&nasm->num_ldata_completed, 1);
}

ssize_t dse_get_ldata_result (nasm_t *nasm, unsigned int ldata_idx, LAYER_PARAMS *order, void** out_ptr)
{
    nasm_ldata_t *ldata = &nasm->ldata_arr[ldata_idx];
    size_t total_size = 0;
    if (ldata->layer->type == YOLO_LAYER)
    {
        void *output = NULL;
        size_t output_size = 0;
        for (int i = 0; i < nasm->num_ldata; i++)
        {
            nasm_ldata_t *targ_ldata = &nasm->ldata_arr[i];
            if (targ_ldata->layer->type == YOLO_LAYER)
            {
                size_t elem_size = targ_ldata->layer->dnn->element_size;
                size_t data_size = targ_ldata->out_mat_dims[OUT_H] * targ_ldata->out_mat_dims[OUT_W] * elem_size;
                output_size += data_size;
            }
        }
        output = calloc (output_size, 1);
        size_t offset = 0;
        size_t batch_num = nasm->batch_size;
        for (int b = 0; b < batch_num; b++)
        {
            for (int i = 0; i < nasm->num_ldata; i++)
            {
                nasm_ldata_t *targ_ldata = &nasm->ldata_arr[i];
                if (targ_ldata->layer->type == YOLO_LAYER)
                {
                    void *data = NULL;
                    ssize_t data_size = get_ldata_output (&data, targ_ldata, order);
                    total_size += data_size;
                    memcpy ((char*)output + offset, (char*)data + data_size*b, data_size);
                    offset += data_size;
                    free (data);
                }
            }
        }
        *out_ptr = output;
        return total_size;
    }
    return get_ldata_output (out_ptr, ldata, order);
}

ssize_t dse_get_nasm_result (nasm_t *nasm, LAYER_PARAMS *order, void** out_ptr)
{
    return dse_get_ldata_result (nasm, nasm->num_ldata - 1, order, out_ptr);
}

void print_dse_info (dse_t *dse)
{
    if (dse == NULL)
    {
        ERROR_PRTF ("Error: Invalid arguments to print_dse_info()");
        assert (0);
    }
    printf ("    Run: %d\n", atomic_load (&dse->run));
    printf ("    Kill: %d\n", atomic_load (&dse->kill));
    printf ("    Target: %p\n", dse->target);
    printf ("Peer Data:\n");
    print_peer_info (dse->my_peer_data);
}
void print_dse_group_info (dse_group_t *dse_group)
{
    if (dse_group == NULL)
    {
        ERROR_PRTF ("Error: Invalid arguments to print_dse_group_info()");
        assert (0);
    }
    printf ("//////////////////////// Printing DSE Info //////////////////////////\n");
    printf ("Peer Data:\n");
    print_peer_info (dse_group->my_peer_data);
    printf ("Num DSEs: %d\n", dse_group->num_dses);
    for (int i = 0; i < dse_group->num_dses; i++)
    {
        printf ("DSE %d:\n", i);
        print_dse_info (&dse_group->dse_arr[i]);
    }
}