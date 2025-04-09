@REM change in host
@REM set project_dir=/home/mct/xm

set project_dir=F:\carlasim\carla\carla\
set PYTHONPATH=.
set PYTHONPATH=%PYTHONPATH%;%project_dir%PythonAPI\carla\dist\carla-0.9.12-py3.8-win-amd64.egg
set PYTHONPATH=%PYTHONPATH%;%project_dir%PythonAPI\carla
set PYTHONPATH=%PYTHONPATH%;%project_dir%PythonAPI


set leaderboard_dir=F:\carlasim\leaderboard
set PYTHONPATH=%PYTHONPATH%;%leaderboard_dir%
set PYTHONPATH=%PYTHONPATH%;%leaderboard_dir%\leaderboard
set PYTHONPATH=%PYTHONPATH%;%leaderboard_dir%\team_code
set PYTHONPATH=%PYTHONPATH%;%leaderboard_dir%\scenario_runner
@REM set PYTHONPATH=%PYTHONPATH%;interfuser

set LEADERBOARD_ROOT=leaderboard
set CHALLENGE_TRACK_CODENAME=SENSORS
set HOST=172.17.0.4
set HOST=172.17.0.1
set PORT=2000
set TM_PORT=8000
set DEBUG_CHALLENGE=0
set REPETITIONS=1
set ROUTES=leaderboard/data/training_routes/routes_town01_short.xml
@REM set TEAM_AGENT=leaderboard/team_code/interfuser_agent.py
set TEAM_AGENT=leaderboard/team_code/auto_pilot.py
@REM set TEAM_CONFIG=leaderboard/team_code/interfuser_config.py
set TEAM_CONFIG=leaderboard/team_code/auto_pilot_config.yaml
set CHECKPOINT_ENDPOINT=leaderboard/results/sample_result.json
set SCENARIOS=leaderboard/data/scenarios/town05_all_scenarios.json
set SAVE_PATH=data/train
@REM # set SAVE_PATH=data/val
set RESUME=True

python %LEADERBOARD_ROOT%/leaderboard/leaderboard_evaluator.py ^
--scenarios=%SCENARIOS%  ^
--routes=%ROUTES% ^
--route-id=%ROUTE_ID% ^
--repetitions=%REPETITIONS% ^
--track=%CHALLENGE_TRACK_CODENAME% ^
--checkpoint=%CHECKPOINT_ENDPOINT% ^
--agent=%TEAM_AGENT% ^
--agent-config=%TEAM_CONFIG% ^
--debug=%DEBUG_CHALLENGE% ^
--record=%RECORD_PATH% ^
--resume=%RESUME% ^
--timeout=600
