from airflow import DAG
from airflow.providers.ssh.operators.ssh import SSHOperator
from datetime import datetime

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2023, 1, 1),
}

dag = DAG('clean_data_10m', default_args=default_args, schedule_interval='*/10 * * * *')

time_now = datetime.now()
time_now = time_now.strftime("%m_%d_%Y_%H_%M")

ssh_task = SSHOperator(
    task_id='test_ssh_command',
    ssh_conn_id='sshfraudcls',
    command=f'python3 /home/ubuntu/data_cleaning_script.py --file_path {time_now}',
    conn_timeout=72000,
    cmd_timeout=72000,
    dag=dag
)

ssh_task

