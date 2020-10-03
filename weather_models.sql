CREATE TABLE IF NOT EXISTS Task_Result.weather_models (
  `id` int(11) NOT NULL,
  `size` int(11) NOT NULL,
  `offset` int(11) NOT NULL,
  `intercept` float NOT NULL,
  `coefficient` float NOT NULL,
  `mean_absolute_error` float NOT NULL,
  `mean_squared_error` float NOT NULL,
  `root_mean_squared_error` float NOT NULL
); 

ALTER TABLE `weather_models`
 ADD PRIMARY KEY (`id`);
