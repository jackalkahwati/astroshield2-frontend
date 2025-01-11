exports.up = function(knex) {
  return knex.schema
    .createTable('spacecraft', function(table) {
      table.string('id').primary();
      table.string('name').notNullable();
      table.string('status').notNullable().defaultTo('operational');
      table.jsonb('configuration');
      table.timestamps(true, true);
    })
    .createTable('telemetry', function(table) {
      table.increments('id').primary();
      table.string('spacecraft_id').references('id').inTable('spacecraft');
      table.timestamp('timestamp').notNullable();
      table.jsonb('measurements').notNullable();
      table.timestamps(true, true);
    })
    .createTable('commands', function(table) {
      table.increments('id').primary();
      table.string('spacecraft_id').references('id').inTable('spacecraft');
      table.string('command_type').notNullable();
      table.jsonb('parameters').notNullable();
      table.string('status').notNullable().defaultTo('pending');
      table.timestamps(true, true);
    })
    .createTable('users', function(table) {
      table.string('id').primary();
      table.string('username').unique().notNullable();
      table.string('password_hash').notNullable();
      table.string('role').notNullable();
      table.timestamps(true, true);
    })
    .createTable('audit_logs', function(table) {
      table.increments('id').primary();
      table.string('user_id').references('id').inTable('users');
      table.string('action').notNullable();
      table.jsonb('details');
      table.timestamp('timestamp').notNullable().defaultTo(knex.fn.now());
    });
};

exports.down = function(knex) {
  return knex.schema
    .dropTableIfExists('audit_logs')
    .dropTableIfExists('users')
    .dropTableIfExists('commands')
    .dropTableIfExists('telemetry')
    .dropTableIfExists('spacecraft');
}; 