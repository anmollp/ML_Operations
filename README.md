**DataSet Description**
>The dataset contains information on weather conditions recorded on each day at various weather stations around the world. Information includes precipitation, snowfall, temperatures, wind speed and whether the day included thunderstorms or other poor weather conditions.

>So our task is to predict the maximum temperature taking input feature as minimum temperature.

1. Setup local mysql and create two databases namely Task, Task_Result respectively, with username and password set to 'root' for the databases.

2. Create table in each database using the table DDL provided in TableSchemas.sql in the folder.

3. Once the setup is complete run the data_load.sh shell script to load data into the Task database.

4. In the `ML_Operations` folder build image using the Dockerfile.
Ex: docker build --tag ml_ops_app . 

5. Run the flask application.
Ex: docker run --name ml_ops_app_v1 -p 5000:5000 ml_ops_app

**Note**: In case mysql local is not accessible, it would be a binding issue. Make the following changes:
In /etc/mysql/my.cnf add the following line -> bind-address    = 0.0.0.0

*Use the curl(s) provide in apis.txt to test.*
