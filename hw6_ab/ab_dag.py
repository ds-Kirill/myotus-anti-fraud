from airflow import DAG
from airflow.providers.ssh.operators.ssh import SSHOperator
from datetime import datetime

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2023, 1, 1),
}

dag = DAG('train_model', default_args=default_args, schedule_interval='0 0 * * 0')

ssh_task = SSHOperator(
    task_id='ssh_command',
    ssh_conn_id='sshconn',
    command=f'python3 /home/ubuntu/validate.py',
    conn_timeout=72000,
    cmd_timeout=72000,
    dag=dag
)

ssh_task