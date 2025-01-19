from confluent_kafka.admin import AdminClient
from confluent_kafka import Producer

if __name__ == '__main__':

    producer = Producer({'bootstrap.servers': '192.168.65.3:32486'})

    for i in range(10):
        print(f'Producing message: {i}')
        producer.produce('mridul', value=f'hello this is rakesh from indian it center')
        producer.flush()
